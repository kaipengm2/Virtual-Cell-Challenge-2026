"""Generate predictions from source statistics and destination controls."""

import argparse
from pathlib import Path
import time
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from model import (
    load_xatlas,
    fuse_source_centered,
    aligned_cd4,
    add_cd4_family,
    desired_mean,
    apply_promoter_prior,
    dual_moment_counts,
    seed_for,
)

SOURCES = [
    "K562_GWPS_CPM_full_statistics.npz",
    "HCT116_full_statistics.npz",
    "HEK293T_full_statistics.npz",
    "H1_2025_full_statistics.npz",
]
WEIGHTS = [2, 1, 1, 2]
SEED = 20260910
CELLS = 400


def axis_csv(path, column):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if column not in table:
        raise ValueError(f"{Path(path).name}: expected a {column!r} column")
    values = table[column].to_numpy(dtype=str)
    if not len(values) or any((not x.strip() for x in values)) or len(set(values)) != len(values):
        raise ValueError(f"{column}: labels must be nonempty and unique")
    if column == "target_gene" and "non-targeting" in values:
        raise ValueError("Prediction panel must not contain controls")
    return values


class CountWriter:
    """Append integer count blocks; retain int64 CSR pointers beyond 2^31 NNZ."""

    def __init__(self, path, targets, genes, contexts, cells):
        self.path = Path(path)
        self.rows = 0
        self.nnz = 0
        self.nobs = len(targets) * len(contexts) * cells
        self.ngenes = len(genes)
        obs = pd.DataFrame(
            {
                "target_gene": np.tile(np.repeat(targets, cells), len(contexts)),
                "context": np.repeat(contexts, len(targets) * cells),
            },
            index=[f"{c}_{t}_{i}" for c in contexts for t in targets for i in range(cells)],
        )
        ad.AnnData(
            sparse.csr_matrix((self.nobs, len(genes)), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=genes),
        ).write_h5ad(self.path)
        self.file = h5py.File(self.path, "r+")
        group = self.file["X"]
        for name in ["data", "indices", "indptr"]:
            del group[name]
        self.data = group.create_dataset(
            "data",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            chunks=(1048576,),
            compression="lzf",
            shuffle=True,
        )
        self.indices = group.create_dataset(
            "indices",
            shape=(0,),
            maxshape=(None,),
            dtype="int32",
            chunks=(1048576,),
            compression="lzf",
            shuffle=True,
        )
        self.pointers = group.create_dataset("indptr", shape=(self.nobs + 1,), dtype="int64")
        self.pointers[0] = 0

    def append(self, counts):
        if (
            counts.ndim != 2
            or counts.shape[1] != self.ngenes
            or self.rows + len(counts) > self.nobs
        ):
            raise ValueError("Unexpected count block shape")
        if (
            not np.isfinite(counts).all()
            or (counts < 0).any()
            or (counts != np.floor(counts)).any()
        ):
            raise ValueError("Expected finite nonnegative integer counts")
        if (counts.sum(axis=1, dtype=np.float64) > 1000000).any():
            raise ValueError("Cell depth exceeds 1,000,000")
        block = sparse.csr_matrix(counts.astype(np.uint32))
        end = self.nnz + block.nnz
        if end > 4750000000:
            raise ValueError("Sparse element limit exceeded")
        self.data.resize((end,))
        self.indices.resize((end,))
        self.data[self.nnz : end] = block.data
        self.indices[self.nnz : end] = block.indices
        self.pointers[self.rows + 1 : self.rows + len(counts) + 1] = (
            block.indptr[1:].astype(np.int64) + self.nnz
        )
        self.rows += len(counts)
        self.nnz = end

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        complete = self.rows == self.nobs and int(self.pointers[-1]) == self.nnz
        self.file.close()
        if kind is None and (not complete):
            raise ValueError("Incomplete prediction output")


def control_template(path, genes, cells, pool_k, seed):
    controls = ad.read_h5ad(path)
    if not np.array_equal(controls.var_names, genes):
        raise ValueError("Control gene order differs from gene_names.csv")
    raw = sparse.csr_matrix(controls.X, dtype=np.float64)
    if (
        not np.isfinite(raw.data).all()
        or (raw.data < 0).any()
        or (raw.data != np.floor(raw.data)).any()
    ):
        raise ValueError("Controls must contain raw nonnegative integer counts")
    library = np.asarray(raw.sum(axis=1)).ravel()
    if (library <= 0).any() or len(library) < cells * pool_k:
        raise ValueError(
            "Need positive-depth controls and at least cells_per_target * pool_k donors"
        )
    mean = np.zeros(len(genes))
    for left in range(0, len(library), 256):
        block = raw[left : left + 256].toarray() / library[left : left + 256, None]
        mean += block.sum(axis=0)
    mean /= len(library)
    bulk = np.asarray(raw.sum(axis=0)).ravel()
    bulk /= bulk.sum()
    selected = np.random.default_rng(seed).choice(len(library), cells * pool_k, replace=False)
    selected = selected[np.argsort(library[selected], kind="stable")]
    template = raw[selected].toarray() / library[selected, None]
    template = template.reshape(cells, pool_k, len(genes)).mean(axis=1)
    depths = np.rint(library[selected].reshape(cells, pool_k).mean(axis=1)).astype(np.int64)
    if depths.min() < 1 or depths.max() > 1000000:
        raise ValueError("Invalid predicted library depths")
    return (template, depths, mean, bulk)


def predict(data_dir, output):
    root, output = Path(data_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    pending = output.with_name(output.stem + ".partial.h5ad")
    if pending.exists():
        raise FileExistsError(pending)
    targets = axis_csv(root / "pert_counts.csv", "target_gene")
    genes = axis_csv(root / "gene_names.csv", "gene_name")
    if (len(targets), len(genes)) != (300, 18533):
        raise ValueError("Expected the official 300-target, 18533-gene panel")
    panel = pd.read_csv(root / "pert_counts.csv")
    if "n_cells" in panel and not (panel.n_cells == CELLS).all():
        raise ValueError("Expected 400 cells per target")
    sources = [load_xatlas(root / name, prior_counts=100000) for name in SOURCES]
    cd4, available = aligned_cd4(
        root / "CD4_DE_statistics.npz", targets, genes, common_subtract=1, center_scope="source"
    )
    base = {
        space: fuse_source_centered(
            sources, WEIGHTS, targets, genes, space=space, common_subtract=1
        )
        for space in ["log2fc", "bulk_delta"]
    }
    del sources
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with CountWriter(pending, targets, genes, list("ABC"), CELLS) as writer:
        for ci, context in enumerate("ABC"):
            template, depths, mean, bulk = control_template(
                root / f"context_{context}.h5ad", genes, CELLS, 4, SEED + ci
            )
            desired = []
            for space, control, amplitude in [("log2fc", mean, 0.6), ("bulk_delta", bulk, 0.3)]:
                effect, coverage = base[space]
                effect, _ = add_cd4_family(
                    effect, coverage, cd4, available, 0.5, space=space, control_probability=control
                )
                probability = desired_mean(
                    control, effect, space=space, amplitude=amplitude, clip=3
                )
                probability, _ = apply_promoter_prior(
                    probability, control, targets, genes, root / "official_pairs.csv", 0.15
                )
                desired.append(probability)
            for ti, target in enumerate(targets):
                counts = dual_moment_counts(
                    template,
                    desired[0][ti],
                    desired[1][ti],
                    depths=depths,
                    seed=seed_for(context + ":" + target, SEED),
                )
                writer.append(counts)
                if (ti + 1) % 50 == 0:
                    print(f"{context}: {ti + 1}/300, {time.monotonic() - started:.0f}s", flush=True)
    pending.rename(output)
    print(f"Saved {output.name}: {writer.nobs} cells, {writer.nnz} nonzeros")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("prediction.h5ad"))
    args = parser.parse_args()
    predict(args.data_dir, args.output)
