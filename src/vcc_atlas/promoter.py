"""A bounded local-promoter mean prior, using coordinates and controls only."""

import numpy as np
import pandas as pd


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
        i, j = ti[row.target], gi[row.neighbor]
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
    return out, changed
