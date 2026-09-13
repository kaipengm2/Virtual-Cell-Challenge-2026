# VCC Atlas Transfer

[中文说明](README.zh-CN.md)

A standalone implementation of **v38 / FullAtlas BalancedCounts**, a multi-atlas
response-transfer baseline for the Virtual Cell Challenge 2026. It transfers
measured perturbation responses to a destination control population and generates
raw integer single-cell counts. Inference needs a CPU and prepared statistics;
there is no neural-network training or foundation-model dependency.

The original public validation submission scored **0.1545618019** and was
**82 / 839 in the snapshot taken on September 10, 2026, 08:34:53 Asia/Shanghai**.
This is a historical public leaderboard observation, not a current rank or a
private-test result. See [historical result and reproduction](docs/REPRODUCIBILITY.md).

The standalone release was checked against the **entire historical prediction**:
all 360,000 cells, 2,437,685,571 nonzero count values, sparse indices, metadata,
and saved profile arrays agree exactly under the frozen inputs. The validation
receipt and its environment are included; no new leaderboard submission was made.

## Quick start: no data download

Use Python 3.11 or newer. Python 3.13 is the locally tested environment.
From this source directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
vcc-atlas demo --output runs/demo
pytest -q
```

The demo generates entirely synthetic inputs and a **144 × 24** integer-count
H5AD through the same inference pipeline. It is an installation and correctness
example, not a challenge submission or an accuracy benchmark. Use a fresh output
directory for each run; existing output directories are never overwritten.

For the versions used in the full historical comparison:

```bash
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
```

## Reproduce the competition prediction

Prepare the files described in [DATA.md](docs/DATA.md), keeping them in a flat
directory such as `data/v38`. Raw and derived biological data are not bundled.

```bash
vcc-atlas verify-inputs \
  --manifest docs/v38-inputs.sha256.json --data-dir data/v38
vcc-atlas predict \
  --config configs/v38.json --data-dir data/v38 --output runs/v38
```

`verify-inputs` checks the frozen **prepared-file** hashes. `predict` also accepts
valid newly extracted statistics; such a run is not automatically an exact v38
reproduction. All paths inside the config resolve against `--data-dir`.

The output contains:

- `prediction.h5ad`: sparse, nonnegative integer-valued counts stored as float32,
  with `target_gene` and `context` in `obs`.
- `effects.npz` and `requested_bulk.npz`: transferred effects, coverage weights,
  and desired arithmetic-mean and pseudobulk probabilities.
- `moment_fit_*.json`: projection, fitting, rounding, and final CPM errors.
- `config.json` and `run.json`: configuration, dependency versions, input/code
  hashes, dimensions, runtime, and output hash. A completed run has `run.json`.

The v38 config emits **3 contexts × 300 targets × 400 cells**, with 18,533 genes.
The original generation took about 608 seconds, excluding source extraction,
compression, official validation, and scoring. Runtime is hardware-dependent.
The prepared five-source NPZ files occupy about 395 MB; the original raw output
occupies about 4.4 GiB. Allow additional memory for loaded statistics, donor
templates, and profile arrays; 8 GB is a starting inference budget, not a measured
peak-memory guarantee.

## Package a submission

The optional lossless compactor improves compression by permuting cells within
each context and writing CSC. It preserves within-target cell order and counts.
The historical compact H5AD was about 892 MiB. For that full matrix the compactor
uses roughly 39 GB of temporary disk, released when it finishes.

```bash
python -m pip install -e '.[compact]'
python scripts/compact_prediction.py \
  runs/v38/prediction.h5ad runs/v38/prediction_compact.h5ad
vcc prep runs/v38/prediction_compact.h5ad \
  -g data/v38/gene_names.csv --perts data/v38/pert_counts.csv \
  -o runs/v38/prediction.vcc
```

Install the official [VCC CLI](https://vcc-cli-wiki.virtualcellchallenge.org/)
separately. Version 0.2.0 was used for the historical submission. For machines
limited by loading the full sparse matrix, [SUBMISSION.md](docs/SUBMISSION.md)
describes the optional disk-mapped adapter. No command in this repository
automatically uploads predictions, authenticates, or starts scoring.

## Method

1. Load source raw-count sums and arithmetic mean CPM with source-matched controls.
2. Shrink noisy source estimates toward controls using 100,000 prior counts.
3. Compute CPM log2 fold changes and pseudobulk log1p(50k) differences separately.
4. Center each source using its **entire prepared target panel**, excluding each
   perturbation's own target coordinate from the centering estimate.
5. Fuse K562, HCT116, HEK293T, and H1 with weights **2 : 1 : 1 : 2**, masking
   unmeasured genes and targets with fewer than 20 cells. Pool the three CD4
   conditions as one family with weight **0.5**.
6. Apply amplitudes **0.6** for CPM and **0.3** for bulk, plus a bounded local
   promoter prior. Pool four depth-neighboring sampled control cells per template.
7. Fit arithmetic CPM and depth-weighted bulk moments jointly, systematically
   round to integer counts, and repair gene totals while preserving each row depth.

[METHOD.md](docs/METHOD.md) specifies units, masks, source panels, and limitations.
The source panel is the panel stored in each prepared NPZ; it is **not every
perturbation measured in the original genome-wide atlas**.

## Scope and limitations

This baseline uses public measurements of some of the same perturbations in other
cell backgrounds. It does not establish generalization to perturbations absent
from every source. Donor pooling and count repair can alter biological variation;
matching means does not establish accurate cell-state distributions or differential
expression. Public leaderboard tuning is not an independent test of generalization.

This repository preserves a useful competition baseline and its numerical
reproduction path. It does not claim that each component independently improves
accuracy, that the method is a new foundation model, or that it beats all such models.

## Layout and license

`src/vcc_atlas/` contains inference; `configs/` contains the frozen v38 config;
`scripts/` contains extraction, compression, and comparison tools; `tests/` uses
synthetic fixtures only. Experimental notebooks, research branches, credentials,
large caches, and competition data are excluded.

Code is provided under [MIT](LICENSE). External datasets, annotations, and software
retain their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md). Data preparation and
submission tools are optional and are not installed by default with the core model.
