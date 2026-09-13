"""Synthetic smoke example, with no biological observations or network access."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .pipeline import predict


def make_demo(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    data = root / "data"
    data.mkdir()
    rng = np.random.default_rng(123)
    genes = np.asarray([f"G{i}" for i in range(24)])
    targets = genes[:3]
    pd.DataFrame({"gene_name": genes}).to_csv(data / "gene_names.csv", index=False)
    pd.DataFrame({"target_gene": targets, "n_cells": 16}).to_csv(
        data / "pert_counts.csv", index=False
    )
    for context in "ABC":
        probabilities = rng.dirichlet(np.ones(len(genes)) * 10)
        raw = np.asarray(
            [rng.multinomial(int(d), probabilities) for d in rng.integers(5000, 20000, 96)]
        )
        ad.AnnData(sparse.csr_matrix(raw), var=pd.DataFrame(index=genes)).write_h5ad(
            data / f"context_{context}.h5ad"
        )
    p = np.ones(len(genes)) / len(genes)
    pert = np.tile(p, (5, 1)) * np.exp(rng.normal(0, 0.03, (5, len(genes))))
    pert /= pert.sum(axis=1, keepdims=True)
    np.savez_compressed(
        data / "synthetic.npz",
        source=np.asarray("synthetic"),
        targets=genes[:5],
        genes=genes,
        n_cells=np.full(5, 100),
        target_count_sums=(pert * 1e6).astype(np.float32),
        target_mean_cpm=(pert * 1e6).astype(np.float32),
        matched_control_probability=np.tile(p, (5, 1)).astype(np.float32),
        matched_control_mean_cpm=np.tile(p * 1e6, (5, 1)).astype(np.float32),
        measured_genes=np.ones(len(genes), dtype=bool),
    )
    config = {
        "schema_version": 1,
        "sources": [{"path": "synthetic.npz", "weight": 1}],
        "cd4_path": None,
        "cd4_weight": 0,
        "promoter_pairs": None,
        "promoter_fraction": 0.15,
        "contexts": list("ABC"),
        "cells_per_target": 16,
        "pool_k": 4,
        "seed": 20260910,
        "prior_counts": 100000,
        "common_subtract": 1.0,
        "cpm_amplitude": 0.6,
        "bulk_amplitude": 0.3,
        "effect_clip": 3.0,
        "expected_targets": 3,
        "expected_genes": 24,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return root / "config.json", data


def run_demo(root):
    config, data = make_demo(root)
    report = predict(config, data, Path(root) / "prediction")
    print(f"Synthetic example complete: {report['shape']}, {report['nnz']} nonzeros")
