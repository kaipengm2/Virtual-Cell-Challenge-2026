# AtlasShift — Virtual Cell Challenge 2026

AtlasShift is a minimal solution for the [Arc Institute Virtual Cell Challenge 2026 (VCC 2026)](https://virtualcellchallenge.org/): zero-shot prediction of single-cell gene expression responses to CRISPRi perturbations across cell contexts.

It transfers measured responses from five public cell atlases, combines them with target-context controls and a promoter-neighbor correction, and generates integer counts matching mean CPM and pseudobulk profiles. CPU inference; no model training.

Historical VCC 2026 public score: **0.1545618019** (September 10, 2026).

```bash
# Python 3.13; install the official VCC CLI separately
pip install -r requirements.txt
python predict.py --data-dir data --output prediction.h5ad
python compact.py prediction.h5ad prediction_compact.h5ad
python pack.py prediction_compact.h5ad --data-dir data --output prediction.vcc
```

Required files in `data/` (not bundled):

```text
gene_names.csv                 # Official gene order
pert_counts.csv                # Official targets; 400 cells per group
context_A.h5ad                 # Official controls
context_B.h5ad
context_C.h5ad
K562_GWPS_CPM_full_statistics.npz
HCT116_full_statistics.npz
HEK293T_full_statistics.npz
H1_2025_full_statistics.npz
CD4_DE_statistics.npz
official_pairs.csv
```

To prepare source statistics from public data:

```bash
pip install slafdb
python prepare.py --data-dir data --raw-dir raw
# Single-source example: python prepare.py --source hct
```

Raw downloads total about 117 GB, plus an X-Atlas scan. URLs and checksums are in `sources.json`. Keep the full prepared source panels; remote X-Atlas updates may change results.

Inference takes about 10 minutes on the tested machine. Compression and packaging need approximately 39 GB and 29 GB of temporary disk, respectively. `pack.py` uses the installed vcc-cli 0.2.0 with all official checks; use `vcc prep` for other versions.

MIT code license. External data retain their own terms.
