"""Estimate shared source response from the full prepared source target panel."""

import numpy as np
from .transfer import aligned_effect


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
    if len(sources) != len(weights) or any(w < 0 for w in weights):
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
    return result.astype(np.float32), denominator.astype(np.float32)
