"""Transfer publisher DESeq2 log2FC with explicit family and missing-gene masks."""

import numpy as np


def aligned_cd4(path, targets, genes, common_subtract=1.0, center_scope="panel"):
    # Use the audited artifact with Unicode string axes and a recorded SHA256.
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
    return effect.astype(np.float32), denominator > 0


def add_cd4_family(effects, coverage, cd4, available, weight, *, space, control_probability):
    if weight < 0:
        raise ValueError("CD4 weight must be nonnegative")
    if space == "log2fc":
        transferred = cd4
    elif space == "bulk_delta":
        # Convert a relative change to the destination's expression-aware
        # log1p(50k) difference before mixing with existing bulk deltas.
        baseline = np.asarray(control_probability) * 50_000
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
    return result, denominator
