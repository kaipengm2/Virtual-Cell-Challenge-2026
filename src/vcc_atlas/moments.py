"""Experimental cell emission with arithmetic CPM and pseudobulk constraints."""

import numpy as np
from .rounding import repair_bulk_totals


def dual_moment_counts(
    template, probability, bulk_probability, *, depths, seed, iterations=100, tolerance=2e-4
):
    template = np.asarray(template)
    probability = np.asarray(probability)
    bulk_probability = np.asarray(bulk_probability)
    raw_depths = np.asarray(depths)
    if template.ndim != 2 or not all(template.shape):
        raise ValueError("Template must be a nonempty cell-by-gene matrix")
    if probability.shape != (template.shape[1],) or bulk_probability.shape != probability.shape:
        raise ValueError("Moment gene axes differ from the template")
    for value in [template, probability, bulk_probability, raw_depths]:
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError("Inputs must be finite and nonnegative")
    if (
        (raw_depths != np.floor(raw_depths)).any()
        or (raw_depths < 1).any()
        or (raw_depths > 1_000_000).any()
    ):
        raise ValueError("Depths must be integers in [1, 1000000]")
    if (template.sum(axis=1) <= 0).any():
        raise ValueError("Template rows must have positive mass")
    if not np.isclose(probability.sum(), 1, rtol=0, atol=1e-8) or not np.isclose(
        bulk_probability.sum(), 1, rtol=0, atol=1e-8
    ):
        raise ValueError("Moment probabilities must sum to one")
    if (
        type(iterations) is not int
        or iterations < 1
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("Invalid fitting iterations or tolerance")
    x = np.asarray(template, dtype=np.float64).copy()
    p = np.asarray(probability, dtype=np.float64)
    bulk = np.asarray(bulk_probability, dtype=np.float64)
    depths = np.broadcast_to(np.asarray(depths, dtype=np.int64), (len(x),))
    if (depths <= 0).any() or (p < 0).any() or (bulk < 0).any():
        raise ValueError("Invalid moment constraints")
    z = depths / depths.mean()
    desired = len(x) * p
    # Finite donor depths constrain each gene's bulk/CPM ratio. Project the
    # requested bulk onto that box and the probability simplex, retaining an
    # explicit report of the change rather than silently ignoring a constraint.
    requested_bulk = bulk.copy()
    lower = (z.min() + 0.01 * (1 - z.min())) * p
    upper = (z.max() - 0.01 * (z.max() - 1)) * p
    lo, hi = 0.0, 1.0
    if np.ptp(depths) == 0:
        # Equal depths force both moments to agree; avoid an unbounded search
        # when the sum of the singleton feasible point rounds below one.
        bulk = p.copy()
    else:
        for _ in range(1024):
            if np.clip(bulk * hi, lower, upper).sum() >= 1:
                break
            hi *= 2
            if not np.isfinite(hi):
                raise ValueError("Bulk support cannot satisfy the feasible moment bounds")
        else:
            raise ValueError("Cannot bracket the bulk projection")
    for _ in range(70):
        mid = (lo + hi) / 2
        if np.clip(bulk * mid, lower, upper).sum() < 1:
            lo = mid
        else:
            hi = mid
    bulk = np.clip(bulk * ((lo + hi) / 2), lower, upper)
    projection_error = float(np.abs(bulk - requested_bulk).sum())
    if projection_error > 0.03:
        raise ValueError(f"Requested bulk projection too large: {projection_error}")
    ratio = np.divide(bulk, p, out=np.ones_like(p), where=p > 0)
    x /= np.maximum(x.sum(axis=1, keepdims=True), 1e-30)
    missing = (x.sum(axis=0) == 0) & (desired > 0)
    x[:, missing] = p[missing]
    x = 0.999 * x + 0.001 * p[None, :]
    for step in range(iterations):
        # Column scaling fixes arithmetic means. The affine tilt has weighted
        # mean one and changes only the depth-weighted mean of each column.
        col = x.sum(axis=0)
        x *= np.divide(desired, col, out=np.zeros_like(col), where=col > 0)
        first = z @ x
        mean = np.divide(first, desired, out=np.ones_like(first), where=desired > 0)
        second = (z * z) @ x
        var = np.maximum(
            np.divide(second, desired, out=np.zeros_like(second), where=desired > 0) - mean * mean,
            0,
        )
        tilt = np.divide(ratio - mean, var, out=np.zeros_like(mean), where=var > 1e-12)
        lower = -0.95 / np.maximum(z.max() - mean, 1e-12)
        upper = 0.95 / np.maximum(mean - z.min(), 1e-12)
        tilt = np.clip(tilt, lower, upper)
        x *= 1 + tilt[None, :] * (z[:, None] - mean[None, :])
        x /= np.maximum(x.sum(axis=1, keepdims=True), 1e-30)
        if step % 5 == 4:
            cpm_error = float(np.abs(x.mean(axis=0) - p).sum())
            bulk_error = float(np.abs(z @ x / len(x) - bulk).sum())
            if max(cpm_error, bulk_error) < tolerance:
                break
    cpm_error = float(np.abs(x.mean(axis=0) - p).sum())
    bulk_error = float(np.abs(z @ x / len(x) - bulk).sum())
    if max(cpm_error, bulk_error) > 1e-3:
        raise ValueError(f"Moment fitting failed: CPM {cpm_error}, bulk {bulk_error}")
    expected = x * depths[:, None]
    integer = np.floor(expected).astype(np.int32)
    fractions = expected - integer
    rng = np.random.default_rng(seed)
    for i in range(len(x)):
        residual = int(depths[i]) - int(integer[i].sum())
        if residual:
            cumulative = np.cumsum(fractions[i])
            cumulative *= residual / cumulative[-1]
            locations = np.searchsorted(
                cumulative, np.arange(residual) + rng.random(), side="right"
            )
            np.add.at(integer[i], locations, 1)
    integer, rounding_report = repair_bulk_totals(integer, expected)
    if (integer < 0).any() or not np.array_equal(integer.sum(axis=1), depths):
        raise AssertionError("Integer emission lost row depths")
    return integer, {
        "integer_bulk_repair": rounding_report,
        "iterations": step + 1,
        "requested_bulk_projection_l1": projection_error,
        "cpm_l1_before_rounding": cpm_error,
        "bulk_l1_before_rounding": bulk_error,
    }
