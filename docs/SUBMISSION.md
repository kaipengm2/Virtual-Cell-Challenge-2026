# Prediction validation and packaging

The core output is a minimal H5AD with a float32 sparse matrix of integer values,
`obs.target_gene`, `obs.context`, and ordered gene symbols in `var_names`. It uses
int64 sparse pointers because the full v38 matrix exceeds 2^31 nonzeros.

## Standard path

```bash
python -m pip install -e '.[compact]'
python scripts/compact_prediction.py runs/v38/prediction.h5ad runs/v38/prediction_compact.h5ad
vcc prep runs/v38/prediction_compact.h5ad \
  -g data/v38/gene_names.csv --perts data/v38/pert_counts.csv \
  -o runs/v38/prediction.vcc
```

The compactor verifies every output sparse array, all row sums, nonzero counts,
and row checksums, plus 256 exact row comparisons. It reorders rows from
context/target/cell to context/cell/target, preserving each target group's internal
order. Compression alone does not change model predictions. It needs approximately
16 bytes of temporary disk per nonzero (about 39 GB at v38 size), plus output space.

`vcc prep` performs the official validation and archive creation. Its peak memory
can exceed the memory needed for inference because the full sparse matrix is loaded.
The CLI is separate from this package; see its [documentation](https://vcc-cli-wiki.virtualcellchallenge.org/).

## Optional disk-mapped adapter for vcc-cli 0.2.0

`scripts/prepare_submission.py` replaces only the input reader of the installed
official prep module with a disk-mapped sparse reader. It calls official
`run_prep` with all default 2026 checks enabled. It rejects unexpected extra
AnnData fields and out-of-range sparse indices. The original reader is restored
on exit. The adapter is pinned to **vcc-cli 0.2.0** because it accesses a Python
module interface, not a version-independent public CLI contract.

Run the script with the Python interpreter in the environment that contains the
official CLI. For example, if you installed that CLI in `.vcc-env`:

```bash
.vcc-env/bin/python scripts/prepare_submission.py runs/v38/prediction_compact.h5ad \
  --genes data/v38/gene_names.csv --perts data/v38/pert_counts.csv \
  --scratch-dir scratch/prep --report runs/v38/prep.json \
  --output runs/v38/prediction.vcc
```

Omit `--output` for validation only. Choose a fresh report path. At full v38 size,
the adapter uses approximately 29 GB of temporary disk for mapped float32 values
and int64 indices, released after prep. Disk mapping reduces resident-array
pressure; it does not impose a hard operating-system memory limit.

Use the standard `vcc prep` for other CLI versions. This repository does not
install or modify the CLI, save account tokens, or upload an archive. A completed
archive can be reviewed and submitted separately through the official CLI.
