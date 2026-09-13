"""Summarize all three released 2025 H1 splits with split-matched controls."""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from prepare_common import verified_asset


def summarize(path):
    data = ad.read_h5ad(path, backed="r")
    try:
        labels = data.obs.target_gene.astype(str).to_numpy()
        groups = np.asarray(sorted(set(labels)), dtype=str)
        ids = pd.Categorical(labels, categories=groups).codes
        n = np.bincount(ids, minlength=len(groups))
        genes = data.var_names.astype(str).to_numpy(dtype=str)
        counts = np.zeros((len(groups), len(genes)), dtype=np.float64)
        cpm = np.zeros_like(counts)
        for left in range(0, len(labels), 4096):
            right = min(left + 4096, len(labels))
            x = sparse.csr_matrix(data.X[left:right], dtype=np.float64)
            lib = np.asarray(x.sum(axis=1)).ravel()
            if (
                (lib <= 0).any()
                or not np.isfinite(x.data).all()
                or (x.data < 0).any()
                or (x.data != np.floor(x.data)).any()
            ):
                raise ValueError("Invalid H1 raw counts")
            membership = sparse.csr_matrix(
                (np.ones(right - left), (ids[left:right], np.arange(right - left))),
                shape=(len(groups), right - left),
            )
            counts += (membership @ x).toarray()
            cpm += (membership @ x.multiply(1e6 / lib[:, None])).toarray()
            if left % (4096 * 8) == 0:
                print(f"{Path(path).name}: {right}/{len(labels)} cells", flush=True)
        return dict(
            targets=groups, genes=genes, n_cells=n, count_sums=counts, mean_cpm=cpm / n[:, None]
        )
    finally:
        data.file.close()


def assemble(summaries):
    targets, counts, means, n, cp, cm, splits = [], [], [], [], [], [], []
    genes = summaries[0]["genes"]
    for label, d in zip(["train", "validation", "test"], summaries, strict=True):
        if not np.array_equal(genes, d["genes"]):
            raise ValueError("H1 gene axes differ")
        controls = np.flatnonzero(d["targets"] == "non-targeting")
        if len(controls) != 1:
            raise ValueError("Expected one control group per H1 split")
        c = int(controls[0])
        ctrl = d["count_sums"][c] / d["count_sums"][c].sum()
        mean = d["mean_cpm"][c]
        if label == "train":
            global_cp, global_cm = ctrl.copy(), mean.copy()
        for i, target in enumerate(d["targets"]):
            if target == "non-targeting":
                continue
            if str(target) in targets:
                raise ValueError("Overlapping H1 treated targets across splits")
            targets.append(str(target))
            counts.append(d["count_sums"][i].copy())
            means.append(d["mean_cpm"][i].copy())
            n.append(int(d["n_cells"][i]))
            cp.append(ctrl.copy())
            cm.append(mean.copy())
            splits.append(label)
    return dict(
        source=np.asarray("H1_2025_public"),
        targets=np.asarray(targets),
        genes=genes,
        n_cells=np.asarray(n),
        target_count_sums=np.asarray(counts, dtype=np.float32),
        target_mean_cpm=np.asarray(means, dtype=np.float32),
        matched_control_probability=np.asarray(cp, dtype=np.float32),
        matched_control_mean_cpm=np.asarray(cm, dtype=np.float32),
        global_control_probability=global_cp.astype(np.float32),
        global_control_mean_cpm=global_cm.astype(np.float32),
        measured_genes=np.ones(len(genes), dtype=bool),
        public_2025_split=np.asarray(splits),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ["train", "validation", "test"]:
        parser.add_argument("--" + split, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, provenance = [], {}
    for split in ["train", "validation", "test"]:
        path = getattr(args, split)
        provenance[split] = verified_asset("h1-" + split, path)
        summary = summarize(path)
        np.savez_compressed(args.output / f"h1_2025_{split}_summary.npz", **summary)
        summaries.append(summary)
    result = assemble(summaries)
    if len(result["targets"]) != 300:
        raise ValueError("Historical H1 source requires 300 distinct targets")
    np.savez_compressed(args.output / "H1_2025_full_statistics.npz", **result)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
