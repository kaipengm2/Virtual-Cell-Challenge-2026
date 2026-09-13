"""Measured response transfer with explicit expression units and missingness."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np


@dataclass
class Source:
    name: str
    targets: np.ndarray
    genes: np.ndarray
    probability: np.ndarray
    control_probability: np.ndarray
    n_cells: np.ndarray
    # Exact per-cell arithmetic means are optional. Bulk-only files cannot supply them.
    mean_cpm: np.ndarray | None = None
    control_mean_cpm: np.ndarray | None = None
    measured: np.ndarray | None = None


def load_xatlas(path: Path, *, matched: bool = True, prior_counts: float = 100_000) -> Source:
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
        # The same prior mass yields a sampling-size shrinkage of the per-cell rate estimate.
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
        return result, mask
    ctrl = source.control_probability
    ctrl = ctrl[ci][None, :] if ctrl.ndim == 1 else ctrl[np.ix_(ri, ci)]
    pert = source.probability[np.ix_(ri, ci)]
    if space == "bulk_delta":
        effect = np.log1p(50_000 * pert) - np.log1p(50_000 * ctrl)
    elif space == "log2fc":
        if source.mean_cpm is None:
            pmean, cmean = 1e6 * pert, 1e6 * ctrl
        else:
            pmean = source.mean_cpm[np.ix_(ri, ci)]
            c = source.control_mean_cpm
            cmean = c[ci][None, :] if c.ndim == 1 else c[np.ix_(ri, ci)]
        effect = np.log2((pmean + pseudocount_cpm) / (cmean + pseudocount_cpm))
    else:
        raise ValueError(space)
    # Remove self-target coordinates from centering estimates, never all panel genes.
    center_values = effect.copy()
    for k, t in enumerate(targets[valid_rows]):
        match = np.flatnonzero(genes[valid_columns] == t)
        center_values[k, match] = np.nan
    common = np.nanmean(center_values, axis=0)
    common = np.nan_to_num(common)
    effect -= common_subtract * common[None, :]
    result[np.ix_(output_rows, output_columns)] = effect
    return result, mask


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
        desired = np.expm1(np.maximum(np.log1p(50_000 * control_probability) + effect, 0))
    else:
        raise ValueError(space)
    total = desired.sum(axis=-1, keepdims=True)
    if (total <= 0).any() or not np.isfinite(desired).all():
        raise ValueError("Invalid desired expression profile")
    return desired / total


def seed_for(target: str, seed: int) -> int:
    return (int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], "little") + seed) % 2**32
