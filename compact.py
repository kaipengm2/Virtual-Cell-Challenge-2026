"""Losslessly permute observations and serialize CSC for smaller uploads.

The input is a context/target/cell ordered prediction with uniform cells per target.
Checks all row checksums, row sums and nnz after conversion, plus exact dense
comparisons for 256 sampled rows. The original prediction is never modified.
"""

import argparse
from pathlib import Path
import tempfile
import time

import anndata as ad
import h5py
import numba as nb
import numpy as np
from scipy import sparse


@nb.njit(cache=True)
def mixed(value):
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


@nb.njit(cache=True)
def scatter_context(
    data,
    indices,
    pointers,
    permutation,
    left,
    right,
    cursors,
    out_data,
    out_rows,
    hashes,
    sums,
    counts,
):
    for new_row in range(left, right):
        old_row = permutation[new_row]
        h = np.uint64(0)
        total = 0.0
        for pos in range(pointers[old_row], pointers[old_row + 1]):
            gene = indices[pos]
            value = data[pos]
            destination = cursors[gene]
            out_data[destination] = value
            out_rows[destination] = new_row
            cursors[gene] += 1
            h += mixed((np.uint64(gene) << np.uint64(32)) | np.uint64(value))
            total += value
        hashes[new_row] = h
        sums[new_row] = total
        counts[new_row] = pointers[old_row + 1] - pointers[old_row]


@nb.njit(cache=True)
def csc_row_checks(data, rows, pointers, n_rows):
    hashes = np.zeros(n_rows, dtype=np.uint64)
    sums = np.zeros(n_rows, dtype=np.float64)
    counts = np.zeros(n_rows, dtype=np.int64)
    for gene in range(len(pointers) - 1):
        last = -1
        for pos in range(pointers[gene], pointers[gene + 1]):
            row = rows[pos]
            value = data[pos]
            if row <= last:
                raise ValueError("CSC row order is not strictly increasing")
            last = row
            hashes[row] += mixed((np.uint64(gene) << np.uint64(32)) | np.uint64(value))
            sums[row] += value
            counts[row] += 1
    return hashes, sums, counts


@nb.njit(parallel=True, cache=True)
def csc_sample(data, rows, pointers, selected):
    output = np.zeros((len(selected), len(pointers) - 1), dtype=np.float32)
    for gene in nb.prange(len(pointers) - 1):
        start = pointers[gene]
        end = pointers[gene + 1]
        column = rows[start:end]
        for i in range(len(selected)):
            pos = np.searchsorted(column, selected[i])
            if pos < len(column) and column[pos] == selected[i]:
                output[i, gene] = data[start + pos]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cells-per-target", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    start = time.time()
    if args.cells_per_target < 1:
        raise ValueError("cells-per-target must be positive")
    nb.set_num_threads(min(args.threads, nb.config.NUMBA_NUM_THREADS))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.stem + ".partial.h5ad")
    if pending.exists():
        raise FileExistsError(pending)
    backed = ad.read_h5ad(args.input, backed="r")
    obs = backed.obs.copy()
    var = backed.var.copy()
    n_rows, n_genes = backed.shape
    backed.file.close()
    contexts = obs.context.astype(str).unique().tolist()
    n_contexts = len(contexts)
    rows_per_context = n_rows // n_contexts
    if rows_per_context % args.cells_per_target:
        raise ValueError("Unexpected cell-count contract")
    n_targets = rows_per_context // args.cells_per_target
    targets = (
        obs.target_gene.astype(str).iloc[: rows_per_context : args.cells_per_target].to_numpy()
    )
    if not np.array_equal(obs.context.astype(str), np.repeat(contexts, rows_per_context)):
        raise ValueError("Input context order differs")
    if not np.array_equal(
        obs.target_gene.astype(str), np.tile(np.repeat(targets, args.cells_per_target), n_contexts)
    ):
        raise ValueError("Input target order differs")
    permutation = (
        np.arange(n_rows)
        .reshape(n_contexts, n_targets, args.cells_per_target)
        .transpose(0, 2, 1)
        .ravel()
    )
    with tempfile.TemporaryDirectory(prefix="repack_", dir=args.output.parent) as temporary:
        temporary = Path(temporary)
        with h5py.File(args.input) as f:
            x = f["X"]
            if x.attrs.get("encoding-type") != "csr_matrix":
                raise ValueError("Expected CSR input")
            pointers = x["indptr"][:].astype(np.int64)
            nnz = int(pointers[-1])
            original = {}
            for key, dtype in [("data", np.float32), ("indices", np.int32)]:
                values = np.memmap(temporary / (key + ".bin"), mode="w+", shape=(nnz,), dtype=dtype)
                for left in range(0, nnz, 4_194_304):
                    values[left : left + 4_194_304] = x[key][left : left + 4_194_304]
                values.flush()
                original[key] = values
                print("Mapped source", key, "seconds", round(time.time() - start), flush=True)
        gene_counts = np.zeros((n_contexts, n_genes), dtype=np.int64)
        for ci in range(n_contexts):
            left = pointers[ci * rows_per_context]
            right = pointers[(ci + 1) * rows_per_context]
            for begin in range(left, right, 4_194_304):
                gene_counts[ci] += np.bincount(
                    original["indices"][begin : min(begin + 4_194_304, right)], minlength=n_genes
                )
        csc_pointers = np.concatenate(([0], np.cumsum(gene_counts.sum(axis=0))))
        if csc_pointers[-1] != nnz:
            raise AssertionError("Gene counts lost entries")
        out_data = np.memmap(temporary / "csc_data.bin", mode="w+", shape=(nnz,), dtype=np.float32)
        out_rows = np.memmap(temporary / "csc_rows.bin", mode="w+", shape=(nnz,), dtype=np.int32)
        expected_hash = np.zeros(n_rows, dtype=np.uint64)
        expected_sum = np.zeros(n_rows)
        expected_nnz = np.zeros(n_rows, dtype=np.int64)
        for ci, context in enumerate(contexts):
            cursor = (csc_pointers[:-1] + gene_counts[:ci].sum(axis=0)).copy()
            end = cursor + gene_counts[ci]
            scatter_context(
                original["data"],
                original["indices"],
                pointers,
                permutation,
                ci * rows_per_context,
                (ci + 1) * rows_per_context,
                cursor,
                out_data,
                out_rows,
                expected_hash,
                expected_sum,
                expected_nnz,
            )
            if not np.array_equal(cursor, end):
                raise AssertionError("CSC cursor mismatch")
            out_data.flush()
            out_rows.flush()
            print("Repacked context", context, "seconds", round(time.time() - start), flush=True)
        actual_hash, actual_sum, actual_nnz = csc_row_checks(
            out_data, out_rows, csc_pointers, n_rows
        )
        np.testing.assert_array_equal(actual_hash, expected_hash)
        np.testing.assert_array_equal(actual_sum, expected_sum)
        np.testing.assert_array_equal(actual_nnz, expected_nnz)
        selected = np.sort(
            np.random.default_rng(20260910).choice(n_rows, min(256, n_rows), replace=False)
        )
        sampled = csc_sample(out_data, out_rows, csc_pointers, selected)
        original_sample = np.zeros_like(sampled)
        for i, new_row in enumerate(selected):
            row = permutation[new_row]
            left = pointers[row]
            right = pointers[row + 1]
            original_sample[i, original["indices"][left:right]] = original["data"][left:right]
        np.testing.assert_array_equal(sampled, original_sample)
        print(
            f"All row checksums, row totals, nnz and {len(selected)} exact row comparisons passed",
            flush=True,
        )
        ad.AnnData(
            sparse.csc_matrix((n_rows, n_genes), dtype=np.float32),
            obs=obs.iloc[permutation],
            var=var,
        ).write_h5ad(pending)
        with h5py.File(pending, "r+") as f:
            x = f["X"]
            for key in ["data", "indices", "indptr"]:
                del x[key]
            for key, values, dtype in [
                ("data", out_data, "float32"),
                ("indices", out_rows, "int64"),
            ]:
                dataset = x.create_dataset(
                    key,
                    shape=(nnz,),
                    dtype=dtype,
                    chunks=(min(1048576, nnz),),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                for left in range(0, nnz, 1048576):
                    dataset[left : left + 1048576] = values[left : left + 1048576]
                print("Wrote compressed", key, "seconds", round(time.time() - start), flush=True)
            x.create_dataset("indptr", data=csc_pointers.astype(np.int64))
        # Re-read stored values to verify the HDF5 filter did not change them.
        with h5py.File(pending) as f:
            for key, expected in [("data", out_data), ("indices", out_rows)]:
                for left in range(0, nnz, 4_194_304):
                    np.testing.assert_array_equal(
                        f["X"][key][left : left + 4_194_304], expected[left : left + 4_194_304]
                    )
    pending.rename(args.output)
    print("Saved " + args.output.name)


if __name__ == "__main__":
    main()
