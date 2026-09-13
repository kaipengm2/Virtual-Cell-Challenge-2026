# Method specification

## Input contract

Rows of each source statistics NPZ are perturbations and columns are gene symbols.
`target_count_sums` contains summed raw counts; `target_mean_cpm` is the arithmetic
mean of per-cell CPM. Matched control probabilities and mean CPM are different
statistics and must not be substituted for one another. String arrays use Unicode
or byte strings, never pickle-backed object arrays. Gene and target axes are unique.

K562 CPM denominators use the full measured library and duplicate symbols are
summed. K562 controls are matched by `gem_group`: target library weights combine
control bulk probabilities; target cell-count weights combine control mean CPM.
X-Atlas uses at most the first 250 controls per sample, sorted by cell integer ID;
it uses source library depths for CPM and the retained gene axis for bulk counts.
H1 uses each released split's own controls. Its global fallback is training controls.

The prepared K562, X-Atlas, and CD4 target panels are the stable ordered union of
the 2026 official panel and the 2025 H1 training targets, excluding non-targeting.
The historical union has 437 entries. H1 combines its 300 distinct public targets
from train, validation, and test. Altering these panels changes centering.

## Transfer

Let `C[t,g]` be source raw-count sums and `q[t,g]` matched control probabilities.
With prior mass `tau = 100000`:

```
source_probability[t] = normalize(C[t] + tau * q[t])
fraction[t] = sum(C[t]) / (sum(C[t]) + tau)
source_mean_cpm[t] = fraction[t] * measured_mean_cpm[t]
                   + (1 - fraction[t]) * control_mean_cpm[t]
```

For a source with at least 20 cells for the target and a measured output gene:

```
cpm_effect  = log2((source_mean_cpm + 1) / (control_mean_cpm + 1))
bulk_effect = log1p(50000 * source_probability)
            - log1p(50000 * control_probability)
```

For each gene, subtract its average effect across valid rows of the **prepared
source target panel**. Exclude the gene's own perturbation row from this average;
do not zero that row's prediction or zero all genes that happen to be targets.
Compute this average before selecting the requested targets, so requesting a
subset does not alter its transferred effect. Intermediate float32 casts and
stable sort orders are retained from v38 for numerical compatibility.

Fuse each target/gene coordinate by a weighted mean over available sources.
Weights are K562=2, HCT116=1, HEK293T=1, H1=2. Missing measurements contribute
neither numerator nor denominator. No-source coordinates receive zero effect.

CD4 contributes publisher DESeq2 log2 fold changes. Require >=20 cells, >=2 guides,
an on-target significant result, and no single-guide, distal-offtarget, or low-target
expression flag. Center each condition independently over its prepared target
panel, then average available conditions. This is **one family of weight 0.5**,
not three independent sources. For bulk-space fusion, convert CD4 log2FC using
the destination bulk baseline before averaging:

```
log1p(50000 * destination_bulk * 2**clip(cd4_log2fc, -10, 10))
- log1p(50000 * destination_bulk)
```

Restore desired destination profiles with CPM amplitude 0.6 and bulk amplitude
0.3. Clip the **scaled** effects to [-3,3]. CPM uses multiplicative `2**effect`;
bulk uses `expm1(max(log1p(50000 * control_bulk) + effect, 0))`. Normalize each
desired profile to unit sum. The two moments remain distinct throughout.

## Promoter prior

GENCODE v47 gene-boundary TSS is start for '+' and end for '-', using 1-based
coordinates. Ambiguous duplicate symbols are excluded. All other measured genes
within 5 kb on the same chromosome are considered, regardless of strand.
The existing relative mean is capped at `control * remaining`, where:

```
ramp = clip(log(max(distance, 500) / 500) / log(10), 0, 1)
remaining = 0.15 + 0.85 * ramp
```

Normalize the row after all caps. The cap is before normalization; the final
normalized coordinate is not guaranteed to remain below that cap. This prior
uses gene boundaries, not inferred active promoters or canonical-transcript TSS.
It is a competition heuristic, not a universal biological law.

## Cell generation

Use seed 20260910 plus context index for donor sampling. Sample 1600 distinct
control cells, stably sort by library depth, and pool each four cells' normalized
profiles to produce 400 templates. A template's depth is the rounded mean donor
depth, preserving depth variation. Destination control mean CPM uses all controls.

Let `x[i,g]` be the fitted normalized cell profile and `z[i]` depth divided by
mean depth. Match both `mean_i x[i,g]` and `mean_i z[i]*x[i,g]`. Project the requested
bulk profile into the depth-induced feasible coordinate bounds and the simplex;
reject projection L1 above 0.03. Alternating column scaling, affine depth tilts,
and row normalization runs at most 100 iterations. Check convergence every five
iterations at L1 2e-4; reject final CPM or bulk L1 above 1e-3.

Multiply by depths, floor counts, and use seeded systematic rounding to preserve
each cell depth. Repair column totals to largest-remainder rounding of the fitted
expected column sums, moving counts only between genes within each cell. Stable
ties preserve deterministic behavior. This repair does **not** guarantee that
every individual cell/gene count is within one count of its expectation.

Reports distinguish requested-versus-projected bulk error, continuous fit errors,
and post-rounding CPM error. Rounding can increase the latter. The seed for a
target uses the first four SHA256 bytes of `context:target` plus the global seed;
it does not depend on Python's randomized `hash()`.

## Interpretation

The method is a weighted measured-response transfer model with nonlinear unit
transforms, priors, and constrained count generation. It is not an end-to-end
trained neural model. The implementation proves a reproducible computation;
component-level benefit needs separate ablations. Validate generalization on
held-out contexts with the destination source removed; never use H1 as a source
when claiming H1 context-held-out performance.
