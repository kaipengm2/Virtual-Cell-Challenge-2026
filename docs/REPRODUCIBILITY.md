# Historical result and reproduction contract

## Historical public result

| Field | Observation |
|---|---|
| Run | `v38_FullAtlas_BalancedCounts` |
| Entry | `rLmt6Rc89MZOjcCYswLn` |
| Submitted | 2026-09-10 00:00:18.787468 UTC |
| Status | Published |
| Overall public score | 0.1545618019323822 |
| Historical rank | 82 / 839 |
| Rank snapshot | 2026-09-10 08:34:53 Asia/Shanghai |
| Matrix shape | 360000 × 18533 |
| Nonzeros | 2437685571 |

The result comes from archived submission status and leaderboard records. No new
submission is implied by this source release. Later v39 scored 0.1502119959,
below this v38 result. The rank above is not a current or final leaderboard claim.

| Raw metric | Historical value |
|---|---:|
| PDS | 0.8077740616871051 |
| MSE | 0.9431524284889559 |
| LFC NMAE | 0.9182856943680399 |
| Direction fidelity yield | 0.507618337809388 |
| Reach | 0.14930146414598824 |
| Significant-gene Jaccard | 0.029995284448905996 |

These are the reported raw metric fields, not uniformly higher-is-better values.
The composite uses the official scaling. Direction fidelity **yield** divides
correct direction calls by `max(n_pred, N_conf)`; a value near 0.5 is not directly
interpretable as random-sign accuracy. This release does not recompute hidden
reference statistics or independently verify server-side metric aggregation.

## Historical artifact identity

| Artifact | SHA256 |
|---|---|
| Original CSR prediction | `30bb4ca2f02c8f350e4b8efccd2b8f3f12ddc3214e2d8b4144e9fe26d64122a9` |
| Original compact CSC prediction | `d47a15187e299a4561a9a75eed4876594d5495fb06461d473313801b11ea7dea` |
| Original submitted archive | `22fec5f2993a862f567950bcd066ce48358f17223a3d4d5f3d6ea341f3917762` |

The original archive contained 889,569,280 bytes. Binary container hashes can
differ with metadata and library versions even when the logical count matrix is
identical. Numerical reproduction is assessed using every saved profile array,
every sparse count entry, indices and row pointers, and observation/gene metadata.

[v38-inputs.sha256.json](v38-inputs.sha256.json) identifies the eleven prepared
model and control input files. [legacy-code.sha256.json](legacy-code.sha256.json)
records the original build/core code hashes for provenance; these are not hashes
of the refactored files. Each new run records its own installed code hashes.

## Verify a local reconstruction

```bash
python scripts/compare_runs.py \
  --reference /path/to/historical-v38-directory \
  --candidate runs/v38 --output runs/v38/comparison.json
```

Both directories must contain `prediction.h5ad`, `effects.npz`, and
`requested_bulk.npz`. The script compares the entire prediction, not a sample.
Its sparse comparison operates in chunks and exits nonzero on any difference.
The metadata comparison includes observation names, target/context assignments,
and gene order. It expects the uncompressed CSR output, before the optional CSC
cell permutation.

## Release verification records

On September 13, 2026, the standalone implementation rebuilt the full prediction
in **596.88 seconds** using the frozen eleven input files. The subsequent
**complete** numerical comparison found:

- All 2,437,685,571 stored count values identical.
- All 2,437,685,571 sparse column indices and 360,001 row pointers identical.
- All ten saved NPZ arrays, including axes, effects, coverage, CPM profiles and
  requested bulk profiles identical.
- Observation identities, context/target assignments, and gene order identical.

See [full-reproduction.json](full-reproduction.json) and
[reproduction-run.json](reproduction-run.json). The reconstructed H5AD SHA256 is
`155e2440aed8add083e2259f288c19e66c77106177d3305235f93c577bc9d51f`;
its container bytes differ from the original, while the logical matrix and
metadata compare exactly. Equality was tested on the recorded macOS / CPython
3.13 environment, not across all operating systems or BLAS implementations.

[release-checks.json](release-checks.json) records 22 passing tests, static checks,
and an isolated wheel-installation smoke run. The CI workflow is included but was
not run on a remote GitHub service. Preparation checks are separately recorded in
[preparation-verification.json](preparation-verification.json): all assembled H1
arrays agree with their historical artifact and the regenerated promoter CSV has
the original SHA256. Synthetic unit tests check cell-count conservation, distinct
CPM/bulk normalization, source-panel invariance, missing measurements, malformed
inputs, and small complete prediction/compaction runs.

Large public K562, X-Atlas and CD4 raw downloads/extractions are **not** rerun in the
release verification. X-Atlas's original remote snapshot was not pinned. Exact
inference reproduction from frozen prepared inputs and end-to-end reconstruction
from a mutable public source are therefore separate claims.

The full reconstructed matrix also passed every default official `vcc-cli 0.2.0`
prep check through the disk-mapped reader (validation only, no upload or archive
creation). See [official-prep.json](official-prep.json): all contexts, target
panels, cell counts, count-space requirements and sparse limits passed, with no
fields dropped or genes reordered. This does not constitute a new server score.

## Changes made during extraction into this repository

- Explicit data/config/output arguments replace local filesystem paths.
- Only the numerical v38 path remains; legacy models, leaderboard experiments,
  notebooks, account automation, and foundation-model experiments are excluded.
- Input validation fails early for malformed moments, controls, and configurations.
- Projection search is bounded; equal-depth templates have an explicit feasible
  bulk profile. These defensive branches do not alter valid historical v38 inputs.
- Outputs use a partial filename until generation completes and refuse to overwrite.
- Integer CSR row pointers remain 64-bit across the 2^31 nonzero boundary.
- Synthetic examples, tests, standalone extraction recipes, and provenance are added.

The scientific claims and limitations are stated in [METHOD.md](METHOD.md).
