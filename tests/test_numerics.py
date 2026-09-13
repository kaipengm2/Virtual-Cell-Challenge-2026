import numpy as np
import pandas as pd
import pytest

from vcc_atlas.cd4 import aligned_cd4, add_cd4_family
from vcc_atlas.fusion import fuse_source_centered
from vcc_atlas.moments import dual_moment_counts
from vcc_atlas.promoter import apply_promoter_prior
from vcc_atlas.rounding import repair_bulk_totals
from vcc_atlas.transfer import Source, desired_mean


def source():
    genes = np.asarray(["a", "b", "c"])
    p = np.asarray([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]], dtype=np.float32)
    return Source(
        "synthetic",
        genes,
        genes,
        p,
        np.full(3, 1 / 3, dtype=np.float32),
        np.full(3, 30),
        p * 1e6,
        np.full(3, 1e6 / 3),
        np.ones(3, bool),
    )


def test_centering_is_invariant_to_requested_targets():
    s = source()
    full, mask = fuse_source_centered(
        [s], [2], s.targets, s.genes, space="log2fc", common_subtract=1
    )
    subset, subset_mask = fuse_source_centered(
        [s], [2], np.asarray(["b", "missing"]), s.genes, space="log2fc", common_subtract=1
    )
    np.testing.assert_array_equal(subset[0], full[1])
    np.testing.assert_array_equal(subset_mask[0], mask[1])
    assert not subset[1].any() and not subset_mask[1].any()


def test_missing_gene_does_not_dilute_measured_source():
    first, second = source(), source()
    second.measured[1] = False
    a, _ = fuse_source_centered([first], [1], first.targets, first.genes, space="log2fc")
    b, cov = fuse_source_centered(
        [first, second], [1, 9], first.targets, first.genes, space="log2fc"
    )
    np.testing.assert_array_equal(a[:, 1], b[:, 1])
    np.testing.assert_array_equal(cov[:, 1], np.ones(3))


def test_zero_effect_preserves_baseline():
    control = np.asarray([0.2, 0.3, 0.5])
    for space in ["log2fc", "bulk_delta"]:
        value = desired_mean(control, np.zeros((2, 3)), space=space, amplitude=0.6, clip=3)
        np.testing.assert_allclose(value, np.tile(control, (2, 1)), atol=1e-15)


def test_cd4_conditions_form_one_family(tmp_path):
    path = tmp_path / "cd4.npz"
    values = np.asarray(
        [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]
    )
    np.savez(
        path,
        targets=np.asarray(["t1", "t2"]),
        genes=np.asarray(["g1", "g2"]),
        log2fc=values,
        available=np.ones((3, 2), bool),
        quality_pass=np.ones((3, 2), bool),
        n_cells=np.full((3, 2), 30),
    )
    effect, mask = aligned_cd4(
        path, np.asarray(["t1"]), np.asarray(["g1", "g2"]), center_scope="source"
    )
    np.testing.assert_array_equal(effect, [[-1.0, -1.0]])
    combined, coverage = add_cd4_family(
        np.zeros((1, 2)),
        np.ones((1, 2)),
        effect,
        mask,
        0.5,
        space="log2fc",
        control_probability=np.ones(2) / 2,
    )
    np.testing.assert_allclose(combined, -1 / 3)
    np.testing.assert_array_equal(coverage, np.full((1, 2), 1.5))


def test_promoter_bound_and_normalization(tmp_path):
    path = tmp_path / "pairs.csv"
    pd.DataFrame({"target": ["t"], "neighbor": ["g"], "distance": [500]}).to_csv(path, index=False)
    out, changes = apply_promoter_prior(
        np.asarray([[0.5, 0.5]]), np.asarray([0.4, 0.6]), ["t"], ["g", "x"], path, 0.15
    )
    np.testing.assert_allclose(out, [[0.06 / 0.56, 0.5 / 0.56]])
    assert changes[0]["after_before_normalization"] == 0.06


def test_repair_preserves_rows_and_largest_remainder_columns():
    expected = np.asarray([[1.4, 0.3, 0.3], [0.4, 1.3, 0.3], [0.4, 0.3, 1.3]])
    initial = np.asarray([[2, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=np.int32)
    result, report = repair_bulk_totals(initial.copy(), expected)
    np.testing.assert_array_equal(result.sum(axis=1), [2, 2, 2])
    np.testing.assert_array_equal(result.sum(axis=0), [2, 2, 2])
    assert result.min() >= 0 and report["counts_moved"] == 2
    assert report["bulk_probability_l1_after"] < report["bulk_probability_l1_before"]


@pytest.mark.parametrize("equal_depth", [False, True])
def test_dual_moment_determinism_and_marginals(equal_depth):
    rng = np.random.default_rng(42)
    depths = np.full(40, 10000) if equal_depth else np.arange(5000, 25000, 500)
    latent = rng.dirichlet(np.full(8, 50.0), len(depths))
    p = latent.mean(axis=0)
    bulk = depths @ latent / depths.sum()
    first, report = dual_moment_counts(latent, p, bulk, depths=depths, seed=12)
    second, _ = dual_moment_counts(latent, p, bulk, depths=depths, seed=12)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=1), depths)
    assert first.min() >= 0
    assert np.abs(first.sum(axis=0) / depths.sum() - bulk).sum() < 3e-4
    assert np.abs((first / depths[:, None]).mean(axis=0) - p).sum() < 3e-4
    assert report["cpm_l1_before_rounding"] < 2e-4


@pytest.mark.parametrize(
    "bad", [np.asarray([np.nan, 0.5]), np.asarray([-0.5, 1.5]), np.asarray([0.1, 0.1])]
)
def test_invalid_moments_fail_before_projection(bad):
    with pytest.raises(ValueError):
        dual_moment_counts(np.ones((2, 2)), bad, np.asarray([0.5, 0.5]), depths=[10, 20], seed=1)


def test_infeasible_bulk_fails_explicitly():
    with pytest.raises(ValueError, match="projection too large"):
        dual_moment_counts(np.ones((2, 2)), [0.5, 0.5], [0.99, 0.01], depths=[100, 100], seed=1)


def test_fractional_depth_is_rejected():
    with pytest.raises(ValueError, match="integers"):
        dual_moment_counts(np.ones((2, 2)), [0.5, 0.5], [0.5, 0.5], depths=[10.1, 20], seed=1)
