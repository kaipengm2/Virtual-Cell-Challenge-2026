"""Extract publisher CD4 DESeq2 log2 fold changes on all development targets."""

import hashlib
import json
from pathlib import Path
import time

URL = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.h5ad"
GUIDE = "https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/data_sharing_readme.md"


def main():
    start = time.time()
    import h5py
    from anndata.io import read_elem
    import numpy as np
    import pandas as pd
    import argparse
    from prepare_common import verified_asset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--official-targets", type=Path, required=True)
    parser.add_argument("--h1-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw
    record = verified_asset("cd4", raw)
    sha256 = record["sha256"]
    size = raw.stat().st_size
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    official = pd.read_csv(args.official_targets).target_gene.astype(str).tolist()
    h1 = pd.read_csv(args.h1_targets).target_gene.astype(str).tolist()
    targets = list(dict.fromkeys(official + [t for t in h1 if t != "non-targeting"]))
    with h5py.File(raw, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        genes = (
            var.gene_name.astype(str).to_numpy(dtype=str)
            if "gene_name" in var
            else var.index.astype(str).to_numpy(dtype=str)
        )
        if len(set(genes)) != len(genes):
            raise ValueError("Duplicate CD4 source symbols need explicit resolution")
        selected = np.flatnonzero(obs.target_contrast_gene_name.astype(str).isin(targets))
        chosen = obs.iloc[selected].copy()
        chosen.to_parquet(out / "CD4_selected_observations.parquet")
        layers = {}
        for name in ["log_fc", "adj_p_value", "lfcSE"]:
            ds = f["layers"][name]
            if not isinstance(ds, h5py.Dataset) or ds.ndim != 2:
                raise ValueError("Expected publisher dense DE layer")
            layers[name] = np.asarray(ds[selected, :], dtype=np.float32)
    conditions = sorted(chosen.culture_condition.astype(str).unique())
    shape = (len(conditions), len(targets), len(genes))
    effects = np.zeros(shape, np.float32)
    q = np.ones(shape, np.float32)
    se = np.zeros(shape, np.float32)
    available = np.zeros(shape[:2], bool)
    quality = np.zeros(shape[:2], bool)
    n_cells = np.zeros(shape[:2], np.int64)
    rows_per_result = np.zeros(shape[:2], np.int32)
    for c, condition in enumerate(conditions):
        for t, target in enumerate(targets):
            rows = np.flatnonzero(
                (chosen.culture_condition.astype(str).to_numpy() == condition)
                & (chosen.target_contrast_gene_name.astype(str).to_numpy() == target)
            )
            if not len(rows):
                continue
            metadata = chosen.iloc[rows]
            values = layers["log_fc"][rows]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite publisher log2FC values")
            effects[c, t] = values.mean(axis=0)
            q[c, t] = layers["adj_p_value"][rows].max(axis=0)
            se[c, t] = layers["lfcSE"][rows].mean(axis=0)
            available[c, t] = True
            rows_per_result[c, t] = len(rows)
            n_cells[c, t] = int(metadata.n_cells_target.min())
            quality[c, t] = bool(
                (
                    (metadata.n_guides >= 2)
                    & (~metadata.single_guide_estimate.astype(bool))
                    & metadata.ontarget_significant.astype(bool)
                    & (~metadata.distal_offtarget_flag.astype(bool))
                    & (~metadata.low_target_gex.astype(bool))
                ).all()
            )
    np.savez_compressed(
        out / "CD4_DE_statistics.npz",
        targets=np.asarray(targets),
        genes=genes,
        conditions=np.asarray(conditions),
        log2fc=effects,
        adjusted_p=q,
        lfcSE=se,
        available=available,
        quality_pass=quality,
        n_cells=n_cells,
        rows_per_result=rows_per_result,
    )
    report = {
        "source_url": URL,
        "source_bytes": size,
        "source_sha256": sha256,
        "publisher_guide": GUIDE,
        "units": "Publisher DESeq2 fitted log2 fold change; not arithmetic per-cell CPM ratios and not bulk log1p(50k) deltas.",
        "targets": len(targets),
        "genes": len(genes),
        "conditions": conditions,
        "available_per_condition": available.sum(axis=1).tolist(),
        "quality_pass_per_condition": quality.sum(axis=1).tolist(),
        "max_rows_per_result": int(rows_per_result.max()),
        "seconds": time.time() - start,
        "output_sha256": hashlib.sha256((out / "CD4_DE_statistics.npz").read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
