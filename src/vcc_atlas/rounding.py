"""Repair integer column totals while preserving every cell's count depth."""

import numpy as np


def repair_bulk_totals(integer, expected):
    expected = np.asarray(expected, dtype=np.float64)
    if integer.shape != expected.shape or integer.ndim != 2:
        raise ValueError("Count and expectation axes differ")
    if not np.isfinite(expected).all() or (expected < 0).any() or (integer < 0).any():
        raise ValueError("Invalid count expectations")
    old = integer.copy()
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
    moved = int(difference[difference > 0].sum())
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
    # Fill large demands first; leaving a large column until the end can
    # concentrate many additions into the last few rows with capacity.
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
    safe_depth = np.maximum(row_totals, 1)
    expected_cpm = (expected / safe_depth[:, None]).mean(axis=0)
    scale = max(float(row_totals.sum()), 1)
    return integer, {
        "counts_moved": moved,
        "fraction_of_counts_moved": moved / scale,
        "max_cell_gene_change": int(np.abs(integer - old).max(initial=0)),
        "bulk_probability_l1_before": float(np.abs(old.sum(axis=0) - columns).sum() / scale),
        "bulk_probability_l1_after": float(np.abs(target - columns).sum() / scale),
        "cpm_probability_l1_before": float(
            np.abs((old / safe_depth[:, None]).mean(axis=0) - expected_cpm).sum()
        ),
        "cpm_probability_l1_after": float(
            np.abs((integer / safe_depth[:, None]).mean(axis=0) - expected_cpm).sum()
        ),
    }
