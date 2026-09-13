"""Extract arithmetic CPM and batch-matched controls from public GWPS cells."""

import json
from pathlib import Path
import time


def main():
    start = time.time()
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    import argparse
    from prepare_common import verified_asset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--official-targets", type=Path, required=True)
    parser.add_argument("--h1-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    path = args.raw
    record = verified_asset("k562", path)
    sha256 = record["sha256"]
    a = ad.read_h5ad(path, backed="r")
    print("SOURCE", a.shape, "OBS", list(a.obs), "VAR", list(a.var), flush=True)
    a.obs.head(8).to_csv(out / "source_obs_head.csv")
    if not {"gene", "gem_group"}.issubset(a.obs.columns):
        raise ValueError("Unexpected label/batch schema")
    labels = a.obs.gene.astype(str).to_numpy()
    batch_codes, batches = pd.factorize(a.obs.gem_group.astype(str))
    official = pd.read_csv(args.official_targets).target_gene.astype(str).tolist()
    h1 = pd.read_csv(args.h1_targets).target_gene.astype(str).tolist()
    targets = list(dict.fromkeys(official + [t for t in h1 if t != "non-targeting"]))
    genes0 = a.var.gene_name.astype(str).to_numpy()
    genes = np.asarray(list(dict.fromkeys(genes0)), dtype=str)
    glookup = {g: i for i, g in enumerate(genes)}
    projection = sparse.csr_matrix(
        (np.ones(len(genes0)), (np.arange(len(genes0)), [glookup[g] for g in genes0])),
        shape=(len(genes0), len(genes)),
    )
    n_targets = len(targets)
    n_batches = len(batches)
    target_lookup = {t: i for i, t in enumerate(targets)}
    row_group = np.asarray([target_lookup.get(t, -1) for t in labels], dtype=np.int32)
    control = labels == "non-targeting"
    row_group[control] = n_targets + batch_codes[control]
    rows = np.flatnonzero(row_group >= 0)
    shape = (n_targets + n_batches, len(genes))
    count_sums = np.zeros(shape, dtype=np.float64)
    cpm_sums = np.zeros(shape, dtype=np.float64)
    group_n = np.bincount(row_group[rows], minlength=shape[0])
    target_batch_n = np.zeros((n_targets, n_batches), dtype=np.int64)
    target_batch_library = np.zeros((n_targets, n_batches), dtype=np.float64)
    source_library = 0.0
    print(
        "SELECTED",
        len(rows),
        "CONTROLS",
        control.sum(),
        "TARGETS",
        n_targets,
        "BATCHES",
        n_batches,
        flush=True,
    )
    for left in range(0, len(rows), 1024):
        selected = rows[left : left + 1024]
        x = sparse.csr_matrix(a.X[selected]).astype(np.float64)
        if (
            not np.isfinite(x.data).all()
            or (x.data < 0).any()
            or not np.equal(x.data, np.floor(x.data)).all()
        ):
            raise ValueError("Noninteger or invalid source counts")
        depth = np.asarray(x.sum(axis=1)).ravel()
        if (depth <= 0).any():
            raise ValueError("Zero-depth cell")
        source_library += depth.sum()
        x = x @ projection
        group = row_group[selected]
        assignment = sparse.csr_matrix(
            (np.ones(len(selected)), (group, np.arange(len(selected)))),
            shape=(shape[0], len(selected)),
        )
        count_sums += (assignment @ x).toarray()
        cpm_sums += (assignment @ x.multiply((1e6 / depth)[:, None])).toarray()
        perturb = group < n_targets
        np.add.at(target_batch_n, (group[perturb], batch_codes[selected][perturb]), 1)
        np.add.at(
            target_batch_library, (group[perturb], batch_codes[selected][perturb]), depth[perturb]
        )
        if left % 20480 == 0:
            print(
                "CELLS",
                left + len(selected),
                "/",
                len(rows),
                "SECONDS",
                round(time.time() - start),
                flush=True,
            )
    np.testing.assert_allclose(count_sums.sum(), source_library, rtol=1e-12)
    valid = group_n > 0
    np.testing.assert_allclose(cpm_sums[valid].sum(axis=1) / group_n[valid], 1e6, rtol=1e-10)
    controls = count_sums[n_targets:]
    control_n = group_n[n_targets:]
    if (control_n == 0).any():
        raise ValueError("Batch lacks control cells")
    control_probability = controls / controls.sum(axis=1, keepdims=True)
    control_cpm = cpm_sums[n_targets:] / control_n[:, None]
    n = group_n[:n_targets]
    counts = count_sums[:n_targets]
    means = np.divide(
        cpm_sums[:n_targets], n[:, None], out=np.zeros_like(counts), where=n[:, None] > 0
    )
    cell_weights = np.divide(
        target_batch_n, n[:, None], out=np.zeros_like(target_batch_library), where=n[:, None] > 0
    )
    lib = target_batch_library.sum(axis=1, keepdims=True)
    library_weights = np.divide(
        target_batch_library, lib, out=np.zeros_like(target_batch_library), where=lib > 0
    )
    global_probability = controls.sum(axis=0) / controls.sum()
    global_mean = cpm_sums[n_targets:].sum(axis=0) / control_n.sum()
    np.savez_compressed(
        out / "K562_GWPS_CPM_full_statistics.npz",
        source=np.asarray("K562_GWPS_CPM"),
        targets=np.asarray(targets),
        genes=genes,
        n_cells=n,
        target_count_sums=counts.astype(np.float32),
        target_mean_cpm=means.astype(np.float32),
        matched_control_probability=(library_weights @ control_probability).astype(np.float32),
        matched_control_mean_cpm=(cell_weights @ control_cpm).astype(np.float32),
        global_control_probability=global_probability.astype(np.float32),
        global_control_mean_cpm=global_mean.astype(np.float32),
        measured_genes=np.ones(len(genes), dtype=bool),
    )
    np.savez_compressed(
        out / "K562_GWPS_batch_qc.npz",
        targets=np.asarray(targets),
        batches=np.asarray(batches, dtype=str),
        target_batch_n=target_batch_n,
        target_batch_library=target_batch_library,
        control_n=control_n,
    )
    a.file.close()
    report = {
        "source": record,
        "source_sha256": sha256,
        "source_shape": list(a.shape),
        "genes": len(genes),
        "selected_cells": len(rows),
        "targets": n_targets,
        "targets_with_20_cells": int((n >= 20).sum()),
        "controls": int(control_n.sum()),
        "batch_column": "gem_group",
        "batches": n_batches,
        "seconds": time.time() - start,
        "units": "Per-cell CPM on full released measured gene axis; duplicate symbols summed.",
        "controls_method": "All available NTC cells; target cell-count weights for CPM and target library weights for bulk probability, averaged before taking logs.",
        "checks": [
            "publisher MD5",
            "integer counts",
            "all count totals preserved",
            "every group mean CPM sums to one million",
        ],
    }
    (out / "provenance.json").write_text(json.dumps(report, indent=2))
    print("GWPS CPM EXTRACTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
