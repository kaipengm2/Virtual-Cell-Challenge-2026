# Data preparation

This release contains code, synthetic examples, and provenance manifests. It does
not redistribute raw or derived biological matrices. Obtain data from the original
providers under their applicable access and reuse terms. No API key is needed for
the public assets below; the official challenge download may require VCC login.

## Inference directory

Place these eleven files directly under `data/v38/`:

| File | Role |
|---|---|
| `gene_names.csv` | Header `gene_name`, official ordered gene axis |
| `pert_counts.csv` | Header `target_gene`; optional `n_cells` must be uniformly 400 |
| `context_A.h5ad`, `context_B.h5ad`, `context_C.h5ad` | Official destination control counts |
| `K562_GWPS_CPM_full_statistics.npz` | K562 matched CPM and bulk statistics |
| `HCT116_full_statistics.npz`, `HEK293T_full_statistics.npz` | X-Atlas statistics |
| `H1_2025_full_statistics.npz` | Three public 2025 H1 splits, each with its own controls |
| `CD4_DE_statistics.npz` | Publisher CD4 effects, availability and quality masks |
| `official_pairs.csv` | GENCODE v47 neighbor pairs for the requested panel |

For existing prepared files, copy or symlink only these files. Their historical
SHA256 values are in [v38-inputs.sha256.json](v38-inputs.sha256.json). Do not narrow
source NPZ target axes to 300 requested targets: the additional rows are used for
source centering. Ordinary inference validates schema and records actual hashes;
the explicit `verify-inputs` command enforces historical file identity.

## Official controls

Use the [official VCC CLI documentation](https://vcc-cli-wiki.virtualcellchallenge.org/):

```bash
mkdir -p data/v38 scratch
vcc datasets download controls -o scratch/vcc_2026_controls.zip
unzip -j scratch/vcc_2026_controls.zip \
  '*gene_names.csv' '*pert_counts.csv' '*context_A.h5ad' '*context_B.h5ad' '*context_C.h5ad' \
  -d data/v38
```

The official metadata files used here have headers. Model inputs must retain the
exact official gene order and raw integer counts. Normalized/log-transformed
controls are not valid substitutes.

## Public raw assets

Install optional preparation dependencies from the repository root:

```bash
python -m pip install -e '.[extract]'
```

[raw-assets.json](raw-assets.json) records URLs, sizes and historical SHA256 values.
H1 object generation IDs are pinned. The downloader explicitly selects one asset,
uses a resumable `.partial` file, verifies its hash, then renames it. It does not
download all sources on installation or delete raw inputs after preparation.

```bash
python scripts/download_asset.py h1-targets --output-dir scratch/raw
python scripts/download_asset.py h1-genes --output-dir scratch/raw
```

Other asset choices are `k562`, `h1-train`, `h1-validation`, `h1-test`, `cd4`, and
`gencode47`. K562 alone is 65.8 GB; the three H1 files total 34.4 GB and CD4 is
16.8 GB. All these raw files together are about 117 GB, before X-Atlas scanning,
extraction outputs, or temporary prediction files. Download only what is needed.

## K562

```bash
python scripts/download_asset.py k562 --output-dir scratch/raw
python scripts/prepare_gwps.py \
  --raw scratch/raw/K562_gwps_raw_singlecell_01.h5ad \
  --official-targets data/v38/pert_counts.csv \
  --h1-targets scratch/raw/pert_counts_Training.csv --output scratch/gwps
cp scratch/gwps/K562_GWPS_CPM_full_statistics.npz data/v38/
```

This scans selected targets and non-targeting cells in chunks, preserves all raw
count totals, and builds separate batch-matched CPM and pooled-count controls.

## HCT116 and HEK293T

```bash
python scripts/prepare_xatlas.py --source HCT116 \
  --official-targets data/v38/pert_counts.csv --official-genes data/v38/gene_names.csv \
  --h1-targets scratch/raw/pert_counts_Training.csv --h1-genes scratch/raw/gene_names.csv \
  --output scratch/hct
python scripts/prepare_xatlas.py --source HEK293T \
  --official-targets data/v38/pert_counts.csv --official-genes data/v38/gene_names.csv \
  --h1-targets scratch/raw/pert_counts_Training.csv --h1-genes scratch/raw/gene_names.csv \
  --output scratch/hek
cp scratch/hct/HCT116_full_statistics.npz scratch/hek/HEK293T_full_statistics.npz data/v38/
```

The default backend is `hf://datasets/slaf-project/X-Atlas-Orion/data`, accessed
through SLAF/Lance. `--atlas-root` accepts a local preserved snapshot or another
compatible root. The historical extraction did not freeze a Hub commit and all
Lance fragments; **a fresh scan of the mutable remote atlas is not guaranteed to
reproduce the historical files**. Use the prepared artifact hashes for exact
historical identity. The extraction records selection counts and scan coverage;
the full scan can be expensive and is optional for users who already have the
prepared statistics. The optional SLAF extraction dependency is not part of the
core inference lock.

## H1

```bash
python scripts/download_asset.py h1-train --output-dir scratch/raw
python scripts/download_asset.py h1-validation --output-dir scratch/raw
python scripts/download_asset.py h1-test --output-dir scratch/raw
python scripts/prepare_h1.py \
  --train scratch/raw/adata_Training.h5ad \
  --validation scratch/raw/adata_Validation.h5ad --test scratch/raw/adata_Test.h5ad \
  --output scratch/h1
cp scratch/h1/H1_2025_full_statistics.npz data/v38/
```

These are the released **2025** data splits, not 2026 hidden treated outcomes.
Use all three splits for this competition transfer configuration. Exclude this
source entirely when evaluating generalization to H1 as a held-out context.

## CD4

```bash
python scripts/download_asset.py cd4 --output-dir scratch/raw
python scripts/prepare_cd4.py --raw scratch/raw/GWCD4i.DE_stats.h5ad \
  --official-targets data/v38/pert_counts.csv \
  --h1-targets scratch/raw/pert_counts_Training.csv --output scratch/cd4
cp scratch/cd4/CD4_DE_statistics.npz data/v38/
```

`log_fc` contains publisher DESeq2 log2 fold changes. Do not interpret it as
arithmetic CPM or a bulk log1p difference. Axes are serialized as non-pickled
strings. The publisher metadata supplies the quality flags used in the model.

## GENCODE v47

```bash
python scripts/download_asset.py gencode47 --output-dir scratch/raw
python scripts/prepare_promoters.py --gtf scratch/raw/gencode.v47.annotation.gtf.gz \
  --official-targets data/v38/pert_counts.csv --official-genes data/v38/gene_names.csv \
  --output data/v38/official_pairs.csv
```

This reproduces the v38 gene-boundary prior. Replacing it with a canonical-TSS
annotation is a model change, not a code cleanup.

## Verification limits

The release validation checks the complete prediction against the historical
prepared inputs. H1 assembly is also compared against all historical summary
arrays, and promoter extraction is checked against the full original GTF.
Synthetic fixtures exercise raw H1 normalization. K562, X-Atlas and CD4 raw
downloads/scans are not rerun as part of this release audit. Their extraction
scripts preserve the original arithmetic and remove notebook-specific paths and
runtime package installations; they remain optional reproduction recipes.

NPZ container bytes can change across library versions even when all arrays are
equal. Use numerical comparisons for re-extracted arrays; never rewrite the
historical manifest merely to make a changed input pass.
