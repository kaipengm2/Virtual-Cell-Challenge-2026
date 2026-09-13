"""v38 response transfer and integer count generation."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class Source:
    name: str
    targets: np.ndarray
    genes: np.ndarray
    probability: np.ndarray
    control_probability: np.ndarray
    n_cells: np.ndarray
    mean_cpm: np.ndarray | None = None
    control_mean_cpm: np.ndarray | None = None
    measured: np.ndarray | None = None


def load_xatlas(path: Path, *, matched: bool = True, prior_counts: float = 100000) -> Source:
    with np.load(path, allow_pickle=False) as d:
        counts = d["target_count_sums"].astype(np.float64)
        ctrl = (
            d["matched_control_probability"] if matched else d["global_control_probability"]
        ).astype(np.float64)
        prior = counts + prior_counts * ctrl
        probability = np.divide(
            prior,
            prior.sum(axis=1, keepdims=True),
            out=np.zeros_like(prior),
            where=prior.sum(axis=1, keepdims=True) > 0,
        )
        fraction = np.divide(
            counts.sum(axis=1),
            counts.sum(axis=1) + prior_counts,
            out=np.zeros(len(counts)),
            where=counts.sum(axis=1) + prior_counts > 0,
        )
        control_cpm = (
            d["matched_control_mean_cpm"] if matched else d["global_control_mean_cpm"]
        ).astype(np.float64)
        mean_cpm = fraction[:, None] * d["target_mean_cpm"] + (1 - fraction[:, None]) * control_cpm
        return Source(
            str(d["source"]),
            d["targets"].astype(str),
            d["genes"].astype(str),
            probability.astype(np.float32),
            ctrl.astype(np.float32),
            d["n_cells"],
            mean_cpm.astype(np.float32),
            control_cpm.astype(np.float32),
            d["measured_genes"],
        )


def aligned_effect(
    source: Source,
    targets: np.ndarray,
    genes: np.ndarray,
    *,
    space: str,
    minimum_cells: int = 20,
    pseudocount_cpm: float = 1,
    common_subtract: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rows = {t: i for i, t in enumerate(source.targets)}
    columns = {g: i for i, g in enumerate(source.genes)}
    found_rows = np.asarray([rows.get(t, -1) for t in targets])
    found_columns = np.asarray([columns.get(g, -1) for g in genes])
    valid_rows = found_rows >= 0
    valid_rows[valid_rows] &= source.n_cells[found_rows[valid_rows]] >= minimum_cells
    valid_columns = found_columns >= 0
    if source.measured is not None:
        valid_columns[valid_columns] &= source.measured[found_columns[valid_columns]]
    result = np.zeros((len(targets), len(genes)), dtype=np.float32)
    mask = valid_rows[:, None] & valid_columns[None, :]
    output_rows = np.flatnonzero(valid_rows)
    output_columns = np.flatnonzero(valid_columns)
    ri = found_rows[valid_rows]
    ci = found_columns[valid_columns]
    if not len(ri) or not len(ci):
        return (result, mask)
    ctrl = source.control_probability
    ctrl = ctrl[ci][None, :] if ctrl.ndim == 1 else ctrl[np.ix_(ri, ci)]
    pert = source.probability[np.ix_(ri, ci)]
    if space == "bulk_delta":
        effect = np.log1p(50000 * pert) - np.log1p(50000 * ctrl)
    elif space == "log2fc":
        if source.mean_cpm is None:
            pmean, cmean = (1000000.0 * pert, 1000000.0 * ctrl)
        else:
            pmean = source.mean_cpm[np.ix_(ri, ci)]
            c = source.control_mean_cpm
            cmean = c[ci][None, :] if c.ndim == 1 else c[np.ix_(ri, ci)]
        effect = np.log2((pmean + pseudocount_cpm) / (cmean + pseudocount_cpm))
    else:
        raise ValueError(space)
    center_values = effect.copy()
    for k, t in enumerate(targets[valid_rows]):
        match = np.flatnonzero(genes[valid_columns] == t)
        center_values[k, match] = np.nan
    common = np.nanmean(center_values, axis=0)
    common = np.nan_to_num(common)
    effect -= common_subtract * common[None, :]
    result[np.ix_(output_rows, output_columns)] = effect
    return (result, mask)


def desired_mean(
    control_probability: np.ndarray,
    effects: np.ndarray,
    *,
    space: str,
    amplitude: float,
    clip: float,
) -> np.ndarray:
    effect = np.clip(amplitude * effects, -clip, clip)
    if space == "log2fc":
        desired = control_probability * np.exp2(effect)
    elif space == "bulk_delta":
        desired = np.expm1(np.maximum(np.log1p(50000 * control_probability) + effect, 0))
    else:
        raise ValueError(space)
    total = desired.sum(axis=-1, keepdims=True)
    if (total <= 0).any() or not np.isfinite(desired).all():
        raise ValueError("Invalid desired expression profile")
    return desired / total


def seed_for(target: str, seed: int) -> int:
    return (int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], "little") + seed) % 2**32


def fuse_source_centered(
    sources,
    weights,
    targets,
    genes,
    *,
    space,
    common_subtract=0,
    minimum_cells=20,
    pseudocount_cpm=1,
):
    if len(sources) != len(weights) or any((w < 0 for w in weights)):
        raise ValueError("Invalid source weights")
    numerator = np.zeros((len(targets), len(genes)), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    for source, weight in zip(sources, weights, strict=True):
        full, mask = aligned_effect(
            source,
            source.targets,
            genes,
            space=space,
            minimum_cells=minimum_cells,
            pseudocount_cpm=pseudocount_cpm,
            common_subtract=common_subtract,
        )
        lookup = {t: i for i, t in enumerate(source.targets)}
        found = np.asarray([lookup.get(t, -1) for t in targets])
        rows = np.flatnonzero(found >= 0)
        numerator[rows] += weight * full[found[rows]]
        denominator[rows] += weight * mask[found[rows]]
    result = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return (result.astype(np.float32), denominator.astype(np.float32))


def aligned_cd4(path, targets, genes, common_subtract=1.0, center_scope="panel"):
    with np.load(path, allow_pickle=False) as data:
        source_targets = data["targets"].astype(str)
        source_genes = data["genes"].astype(str)
        values = data["log2fc"].astype(float)
        available = data["available"] & data["quality_pass"] & (data["n_cells"] >= 20)
    target_index = {t: i for i, t in enumerate(source_targets)}
    gene_index = {g: i for i, g in enumerate(source_genes)}
    ri = np.asarray([target_index.get(t, -1) for t in targets])
    ci = np.asarray([gene_index.get(g, -1) for g in genes])
    rows = np.flatnonzero(ri >= 0)
    columns = np.flatnonzero(ci >= 0)
    numerator = np.zeros((len(targets), len(genes)))
    denominator = np.zeros_like(numerator)
    for condition in range(len(values)):
        effect = values[condition][np.ix_(ri[rows], ci[columns])].copy()
        mask = available[condition, ri[rows], None] & np.isfinite(effect)
        if center_scope == "source":
            centering = np.where(available[condition, :, None], values[condition], np.nan)
            for local, target in enumerate(source_targets):
                if target in gene_index:
                    centering[local, gene_index[target]] = np.nan
            count = np.isfinite(centering).sum(axis=0)
            full_common = np.divide(
                np.nansum(centering, axis=0),
                count,
                out=np.zeros(len(source_genes)),
                where=count > 0,
            )
            common = full_common[ci[columns]]
        elif center_scope == "panel":
            centering = np.where(mask, effect, np.nan)
            for local, row in enumerate(rows):
                centering[local, genes[columns] == targets[row]] = np.nan
            count = np.isfinite(centering).sum(axis=0)
            common = np.divide(
                np.nansum(centering, axis=0), count, out=np.zeros(len(columns)), where=count > 0
            )
        else:
            raise ValueError(center_scope)
        effect -= common_subtract * common[None, :]
        numerator[np.ix_(rows, columns)] += np.where(mask, effect, 0)
        denominator[np.ix_(rows, columns)] += mask
    effect = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return (effect.astype(np.float32), denominator > 0)


def add_cd4_family(effects, coverage, cd4, available, weight, *, space, control_probability):
    if weight < 0:
        raise ValueError("CD4 weight must be nonnegative")
    if space == "log2fc":
        transferred = cd4
    elif space == "bulk_delta":
        baseline = np.asarray(control_probability) * 50000
        transferred = np.log1p(baseline[None, :] * np.exp2(np.clip(cd4, -10, 10))) - np.log1p(
            baseline[None, :]
        )
    else:
        raise ValueError(space)
    denominator = coverage + weight * available
    result = np.divide(
        effects * coverage + weight * np.where(available, transferred, 0),
        denominator,
        out=np.zeros_like(effects),
        where=denominator > 0,
    )
    return (result, denominator)


def apply_promoter_prior(probability, control, targets, genes, pairs_path, fraction=0.15):
    if not 0 < fraction <= 1:
        raise ValueError("Invalid remaining expression fraction")
    pairs = pd.read_csv(pairs_path, usecols=["target", "neighbor", "distance"])
    ti = {str(t): i for i, t in enumerate(targets)}
    gi = {str(g): i for i, g in enumerate(genes)}
    out = np.asarray(probability, dtype=np.float64).copy()
    changed = []
    for row in pairs.itertuples(index=False):
        if row.target not in ti or row.neighbor not in gi or row.target == row.neighbor:
            continue
        if not 0 <= row.distance <= 5000:
            raise ValueError("Unexpected promoter distance")
        i, j = (ti[row.target], gi[row.neighbor])
        ramp = np.clip(np.log(max(float(row.distance), 500) / 500) / np.log(10), 0, 1)
        remaining = fraction + (1 - fraction) * ramp
        ceiling = control[j] * remaining
        if out[i, j] > ceiling:
            changed.append(
                {
                    "target": row.target,
                    "neighbor": row.neighbor,
                    "distance": float(row.distance),
                    "remaining_fraction": float(remaining),
                    "before": float(out[i, j]),
                    "after_before_normalization": float(ceiling),
                }
            )
            out[i, j] = ceiling
    out /= out.sum(axis=1, keepdims=True)
    return (out, changed)


def repair_bulk_totals(integer, expected):
    expected = np.asarray(expected, dtype=np.float64)
    if integer.shape != expected.shape or integer.ndim != 2:
        raise ValueError("Count and expectation axes differ")
    if not np.isfinite(expected).all() or (expected < 0).any() or (integer < 0).any():
        raise ValueError("Invalid count expectations")
    row_totals = integer.sum(axis=1, dtype=np.int64)
    columns = expected.sum(axis=0)
    target = np.floor(columns).astype(np.int64)
    remaining = int(row_totals.sum() - target.sum())
    if remaining < 0 or remaining > len(target):
        raise ValueError("Expected and integer grand totals differ")
    if remaining:
        chosen = np.argsort(columns - target, kind="stable")[-remaining:]
        target[chosen] += 1
    difference = integer.sum(axis=0, dtype=np.int64) - target
    capacity = np.zeros(len(integer), dtype=np.int64)
    for gene in np.flatnonzero(difference > 0):
        excess = int(difference[gene])
        rounded_up = integer[:, gene] - np.floor(expected[:, gene]).astype(np.int64)
        rows = np.flatnonzero(rounded_up > 0)
        if rounded_up[rows].sum() < excess:
            raise ValueError("Initial rounding fell below its integer floor")
        rows = rows[np.argsort(-(integer[rows, gene] - expected[rows, gene]), kind="stable")]
        if excess <= len(rows):
            selected = rows[:excess]
            integer[selected, gene] -= 1
            capacity[selected] += 1
        else:
            for row in rows:
                take = min(excess, int(rounded_up[row]))
                integer[row, gene] -= take
                capacity[row] += take
                excess -= take
                if not excess:
                    break
    deficit_genes = np.flatnonzero(difference < 0)
    deficit_genes = deficit_genes[np.argsort(difference[deficit_genes], kind="stable")]
    for gene in deficit_genes:
        deficit = int(-difference[gene])
        while deficit:
            rows = np.flatnonzero(capacity > 0)
            if not len(rows):
                raise AssertionError("Column repair exhausted row capacity")
            take = min(deficit, len(rows))
            score = expected[rows, gene] - integer[rows, gene]
            selected = rows[np.argsort(-score, kind="stable")[:take]]
            integer[selected, gene] += 1
            capacity[selected] -= 1
            deficit -= take
    if capacity.any() or (integer < 0).any():
        raise AssertionError("Column repair lost counts")
    if not np.array_equal(integer.sum(axis=1, dtype=np.int64), row_totals):
        raise AssertionError("Column repair changed a cell depth")
    if not np.array_equal(integer.sum(axis=0, dtype=np.int64), target):
        raise AssertionError("Column repair failed its bulk totals")
    return integer


def dual_moment_counts(
    template, probability, bulk_probability, *, depths, seed, iterations=100, tolerance=0.0002
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
        or (raw_depths > 1000000).any()
    ):
        raise ValueError("Depths must be integers in [1, 1000000]")
    if (template.sum(axis=1) <= 0).any():
        raise ValueError("Template rows must have positive mass")
    if not np.isclose(probability.sum(), 1, rtol=0, atol=1e-08) or not np.isclose(
        bulk_probability.sum(), 1, rtol=0, atol=1e-08
    ):
        raise ValueError("Moment probabilities must sum to one")
    if (
        type(iterations) is not int
        or iterations < 1
        or (not np.isfinite(tolerance))
        or (tolerance <= 0)
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
    requested_bulk = bulk.copy()
    lower = (z.min() + 0.01 * (1 - z.min())) * p
    upper = (z.max() - 0.01 * (z.max() - 1)) * p
    lo, hi = (0.0, 1.0)
    if np.ptp(depths) == 0:
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
        col = x.sum(axis=0)
        x *= np.divide(desired, col, out=np.zeros_like(col), where=col > 0)
        first = z @ x
        mean = np.divide(first, desired, out=np.ones_like(first), where=desired > 0)
        second = z * z @ x
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
    if max(cpm_error, bulk_error) > 0.001:
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
    integer = repair_bulk_totals(integer, expected)
    if (integer < 0).any() or not np.array_equal(integer.sum(axis=1), depths):
        raise AssertionError("Integer emission lost row depths")
    return integer
