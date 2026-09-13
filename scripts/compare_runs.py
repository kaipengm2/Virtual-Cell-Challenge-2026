"""Compare every saved profile and CSR count entry using bounded-memory reads."""

import argparse
import json
from pathlib import Path
import time

import anndata as ad
import h5py
import numpy as np

from vcc_atlas.io import digest


def compare(reference, candidate):
    started = time.monotonic()
    report = {"profile_arrays": {}, "count_arrays": {}}
    for name in ["effects.npz", "requested_bulk.npz"]:
        with (
            np.load(reference / name, allow_pickle=False) as left,
            np.load(candidate / name, allow_pickle=False) as right,
        ):
            if set(left.files) != set(right.files):
                raise ValueError(f"{name}: field names differ")
            for key in left.files:
                a, b = left[key], right[key]
                exact = a.shape == b.shape and np.array_equal(a, b)
                entry = {"shape": list(a.shape), "exact": bool(exact)}
                if a.dtype.kind in "fiu" and a.shape == b.shape:
                    entry["max_absolute_difference"] = float(
                        np.max(np.abs(a.astype(float) - b.astype(float)), initial=0)
                    )
                report["profile_arrays"][f"{name}/{key}"] = entry
    paths = [p / "prediction.h5ad" for p in [reference, candidate]]
    a, b = [ad.read_h5ad(p, backed="r") for p in paths]
    try:
        report["shape"] = list(a.shape)
        report["metadata_exact"] = bool(
            a.shape == b.shape
            and np.array_equal(a.obs_names, b.obs_names)
            and np.array_equal(a.var_names, b.var_names)
            and set(a.obs) == set(b.obs)
            and all(np.array_equal(a.obs[c].astype(str), b.obs[c].astype(str)) for c in a.obs)
        )
    finally:
        a.file.close()
        b.file.close()
    with h5py.File(paths[0]) as left, h5py.File(paths[1]) as right:
        for name in ["indptr", "indices", "data"]:
            x, y = left[f"X/{name}"], right[f"X/{name}"]
            different = 0
            if x.shape != y.shape:
                report["count_arrays"][name] = {"exact": False, "shape_mismatch": True}
                continue
            for start in range(0, len(x), 4 * 1024**2):
                different += int(
                    np.count_nonzero(
                        x[start : start + 4 * 1024**2] != y[start : start + 4 * 1024**2]
                    )
                )
            report["count_arrays"][name] = {
                "elements": len(x),
                "different": different,
                "exact": different == 0,
            }
            print(f"Compared all {name}: {len(x)} entries, {different} differences", flush=True)
    report["reference_prediction_sha256"] = digest(paths[0])
    report["candidate_prediction_sha256"] = digest(paths[1])
    report["exact"] = report["metadata_exact"] and all(
        v["exact"] for k in ["profile_arrays", "count_arrays"] for v in report[k].values()
    )
    report["seconds"] = time.monotonic() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
