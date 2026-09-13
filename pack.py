"""Package counts with the installed VCC CLI, using disk-mapped sparse arrays."""

import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import sys
import tempfile
import anndata as ad
import h5py
import numpy as np
from scipy import sparse


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
    if backed.raw is not None or any(
        (
            any((k is not None for k in getattr(backed, key).keys()))
            for key in ["layers", "obsm", "obsp", "varm", "varp", "uns"]
        )
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
            for left in range(0, len(dataset), 4194304):
                right = min(left + 4194304, len(dataset))
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
    if not all((np.shares_memory(getattr(matrix, key), arrays[key]) for key in arrays)):
        raise AssertionError("Sparse constructor unexpectedly copied disk-mapped arrays")
    return ad.AnnData(matrix, obs=obs, var=var)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("prediction.vcc"))
    parser.add_argument("--scratch-dir", type=Path, default=Path("scratch"))
    args = parser.parse_args()
    if importlib.util.find_spec("vcc") is None:
        cli = shutil.which("vcc")
        if not cli:
            raise SystemExit("Install the official VCC CLI first.")
        with open(cli, "rb") as handle:
            interpreter = handle.readline().decode().strip().removeprefix("#!")
        if (
            not Path(interpreter).is_file()
            or Path(interpreter).absolute() == Path(sys.executable).absolute()
        ):
            raise SystemExit("Run pack.py with the Python interpreter containing vcc-cli 0.2.0.")
        os.execv(interpreter, [interpreter, str(Path(__file__).resolve()), *sys.argv[1:]])
    import vcc.prep as prep

    if version("vcc-cli") != "0.2.0":
        raise SystemExit("This adapter requires vcc-cli 0.2.0; use vcc prep for other versions.")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    expected = digest(args.prediction)
    original_reader = prep.read_h5ad
    with tempfile.TemporaryDirectory(prefix="vcc_", dir=args.scratch_dir) as temporary:

        def reader(path):
            if Path(path).resolve() != args.prediction.resolve():
                raise ValueError("Unexpected prediction input")
            return mapped_adata(path, temporary)

        prep.read_h5ad = reader
        try:
            prep.run_prep(
                input_path=str(args.prediction),
                output_path=str(args.output),
                genes_path=str(args.data_dir / "gene_names.csv"),
                perts_path=str(args.data_dir / "pert_counts.csv"),
            )
        finally:
            prep.read_h5ad = original_reader
    if digest(args.prediction) != expected:
        raise ValueError("Prediction changed during packaging")
    print("Saved " + args.output.name)


if __name__ == "__main__":
    main()
