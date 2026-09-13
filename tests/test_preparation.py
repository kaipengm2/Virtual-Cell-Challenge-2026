"""Small raw-data fixtures validate the independently usable H1 preparation path."""

import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def load_script(name):
    import sys

    directory = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_h1_raw_counts_preserve_distinct_cpm_and_bulk_means(tmp_path):
    module = load_script("prepare_h1")
    counts = np.asarray([[9, 1], [1, 1], [2, 6], [3, 1]])
    obs = pd.DataFrame(
        {"target_gene": ["t", "t", "non-targeting", "non-targeting"]},
        index=["c1", "c2", "c3", "c4"],
    )
    path = tmp_path / "source.h5ad"
    ad.AnnData(sparse.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=["g1", "g2"])).write_h5ad(
        path
    )
    summary = module.summarize(path)
    np.testing.assert_array_equal(summary["targets"], ["non-targeting", "t"])
    np.testing.assert_array_equal(summary["count_sums"], [[5, 7], [10, 2]])
    np.testing.assert_allclose(summary["mean_cpm"], [[500000, 500000], [700000, 300000]])
    summaries = []
    for target in ["a", "b", "c"]:
        copied = {k: v.copy() for k, v in summary.items()}
        copied["targets"] = np.asarray(["non-targeting", target])
        summaries.append(copied)
    result = module.assemble(summaries)
    np.testing.assert_allclose(
        result["matched_control_probability"], np.tile([5 / 12, 7 / 12], (3, 1))
    )
    np.testing.assert_array_equal(result["public_2025_split"], ["train", "validation", "test"])


def test_promoter_preparation_abstains_on_ambiguous_symbols():
    module = load_script("prepare_promoters")
    table = pd.DataFrame(
        {
            "gene": ["t", "near", "far", "ambiguous", "ambiguous"],
            "chromosome": ["chr1"] * 5,
            "strand": ["+", "-", "+", "+", "-"],
            "tss": [1000, 1500, 9000, 1300, 1400],
        }
    )
    pairs = module.pairs(table, ["t"], ["t", "near", "far", "ambiguous"])
    assert list(pairs.neighbor) == ["near"] and pairs.distance.iloc[0] == 500
