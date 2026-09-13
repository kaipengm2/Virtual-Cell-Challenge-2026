"""Run every official prep check using disk-mapped sparse input to bound RAM.

Only loading changes. Validation flags, values, metadata and official prep functions
are unchanged. This prevents a transient int32-to-int64 index copy above 2^31 nnz.
Run with the installed vcc-cli interpreter, not the project environment.
"""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import time

import anndata as ad
import h5py
import numpy as np
from scipy import sparse
import vcc.prep as prep


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mapped_adata(path, temporary):
    backed = ad.read_h5ad(path, backed="r")
    obs = backed.obs.copy()
    var = backed.var.copy()
    shape = backed.shape
    # AnnData 0.12+ may expose its implicit X layer as a None key. Official prep
    # excludes that key from its droppable-field check for the same reason.
    if backed.raw is not None or any(
        any(k is not None for k in getattr(backed, key).keys())
        for key in ["layers", "obsm", "obsp", "varm", "varp", "uns"]
    ):
        raise ValueError("Mapped validator currently supports only minimal prediction files")
    backed.file.close()
    arrays = {}
    with h5py.File(path, "r") as f:
        group = f["X"]
        encoding = group.attrs.get("encoding-type")
        if encoding not in {"csr_matrix", "csc_matrix"}:
            raise ValueError("Only CSR/CSC inputs supported")
        index_bound = shape[1] if encoding == "csr_matrix" else shape[0]
        index_type = np.int64 if len(group["data"]) > np.iinfo(np.int32).max else np.int32
        for key, dtype in [
            ("data", group["data"].dtype),
            ("indices", index_type),
            ("indptr", index_type),
        ]:
            dataset = group[key]
            array = np.memmap(
                Path(temporary) / (key + ".bin"), mode="w+", shape=dataset.shape, dtype=dtype
            )
            for left in range(0, len(dataset), 4_194_304):
                right = min(left + 4_194_304, len(dataset))
                values = dataset[left:right]
                if key == "indices" and ((values < 0).any() or (values >= index_bound).any()):
                    raise ValueError("Sparse index outside the declared axis")
                array[left:right] = values
            array.flush()
            arrays[key] = array
            print("Mapped", key, len(array), "entries", flush=True)
    if (
        arrays["indptr"][0] != 0
        or arrays["indptr"][-1] != len(arrays["data"])
        or (np.diff(arrays["indptr"]) < 0).any()
    ):
        raise ValueError("Invalid sparse pointers")
    constructor = sparse.csr_matrix if encoding == "csr_matrix" else sparse.csc_matrix
    matrix = constructor(
        (arrays["data"], arrays["indices"], arrays["indptr"]), shape=shape, copy=False
    )
    if not all(np.shares_memory(getattr(matrix, key), arrays[key]) for key in arrays):
        raise AssertionError("Sparse constructor unexpectedly copied disk-mapped arrays")
    return ad.AnnData(matrix, obs=obs, var=var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--perts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="Write a .vcc archive; omit for validation only"
    )
    args = parser.parse_args()
    start = time.time()
    if version("vcc-cli") != "0.2.0":
        raise ValueError(
            "This optional adapter is audited for vcc-cli 0.2.0 only; use the standard vcc prep CLI for other versions"
        )
    if args.report.exists():
        raise FileExistsError(args.report)
    expected = digest(args.prediction)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    original_reader = prep.read_h5ad
    with tempfile.TemporaryDirectory(prefix="vcc_validate_", dir=args.scratch_dir) as temporary:

        def reader(path):
            if Path(path).resolve() != args.prediction.resolve():
                raise ValueError("Unexpected validation input")
            return mapped_adata(path, temporary)

        prep.read_h5ad = reader
        try:
            result = prep.run_prep(
                input_path=str(args.prediction),
                genes_path=str(args.genes),
                perts_path=str(args.perts),
                dry_run=args.output is None,
                output_path=str(args.output) if args.output else None,
            )
        finally:
            prep.read_h5ad = original_reader
    if digest(args.prediction) != expected:
        raise ValueError("Input changed while validating")
    official_report = asdict(result)
    official_report["input"] = args.prediction.name
    official_report["output"] = args.output.name if args.output else None
    report = {
        "official_prep": official_report,
        "vcc_cli_version": version("vcc-cli"),
        "prediction_sha256": expected,
        "reader": "disk-mapped original sparse arrays, values unchanged",
        "checks": "all official default prep checks, no disabled validation flags",
        "seconds": time.time() - start,
        "script_sha256": digest(__file__),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
