"""Full-gene XAtlas sufficient statistics for direct count/rate transfer."""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = "hf://datasets/slaf-project/X-Atlas-Orion/data"
CONTROL = "Non-Targeting"


def main():
    start_time = time.time()
    import lance
    import numpy as np
    import pandas as pd
    from slaf import SLAFArray

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["HCT116", "HEK293T"], required=True)
    parser.add_argument("--atlas-root", default=ROOT, help="Local or remote X-Atlas data root")
    parser.add_argument("--official-targets", type=Path, required=True)
    parser.add_argument("--official-genes", type=Path, required=True)
    parser.add_argument("--h1-targets", type=Path, required=True)
    parser.add_argument("--h1-genes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    SOURCE = args.source
    atlas_root = args.atlas_root
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    official_targets = pd.read_csv(args.official_targets)["target_gene"].astype(str).tolist()
    official_genes = pd.read_csv(args.official_genes)["gene_name"].astype(str).tolist()
    h1_counts = pd.read_csv(args.h1_targets)
    h1_genes = pd.read_csv(args.h1_genes)
    h1_targets = h1_counts["target_gene"].astype(str).tolist()
    gene_column = "gene_name" if "gene_name" in h1_genes else h1_genes.columns[0]
    genes = list(dict.fromkeys(official_genes + h1_genes[gene_column].astype(str).tolist()))
    targets = list(
        dict.fromkeys(official_targets + [t for t in h1_targets if t != "non-targeting"])
    )
    atlas = SLAFArray(f"{atlas_root}/{SOURCE}")
    selected_labels = ", ".join(
        "'" + label.replace("'", "''") + "'" for label in targets + [CONTROL]
    )
    cells = atlas.query(
        f"SELECT cell_integer_id, cell_start_index, n_genes_by_counts, total_counts, sample, gene_target FROM cells WHERE gene_target IN ({selected_labels})"
    ).to_pandas()
    cells["gene_target"] = cells["gene_target"].astype(str)
    controls = (
        cells[cells.gene_target == CONTROL]
        .sort_values(["sample", "cell_integer_id"])
        .groupby("sample", observed=True)
        .head(250)
    )
    cells = (
        pd.concat([cells[cells.gene_target != CONTROL], controls])
        .sort_values("cell_integer_id")
        .reset_index(drop=True)
    )
    var = atlas.query("SELECT gene_id, gene_integer_id FROM genes").to_pandas()
    gene_lookup = {g: i for i, g in enumerate(genes)}
    gene_map = np.full(int(var.gene_integer_id.max()) + 1, -1, dtype=np.int32)
    for g, i in zip(var.gene_id.astype(str), var.gene_integer_id, strict=True):
        gene_map[int(i)] = gene_lookup.get(g, -1)
    keys = list(dict.fromkeys(zip(cells["sample"].astype(str), cells.gene_target, strict=True)))
    key_lookup = {key: i for i, key in enumerate(keys)}
    group_ids = np.asarray(
        [
            key_lookup[key]
            for key in zip(cells["sample"].astype(str), cells.gene_target, strict=True)
        ],
        dtype=np.int32,
    )
    group_n = np.bincount(group_ids, minlength=len(keys))
    max_id = int(cells.cell_integer_id.max())
    lookup = np.full(max_id + 1, -1, dtype=np.int32)
    total_lookup = np.zeros(max_id + 1, dtype=np.float64)
    ids = cells.cell_integer_id.to_numpy(dtype=np.int64)
    lookup[ids] = group_ids
    total_lookup[ids] = cells.total_counts.to_numpy(dtype=np.float64)
    count_sums = np.zeros((len(keys), len(genes)), dtype=np.float64)
    cpm_sums = np.zeros_like(count_sums)
    print(
        "selection",
        SOURCE,
        len(targets),
        len(genes),
        len(cells),
        len(keys),
        "accumulator GB",
        (count_sums.nbytes + cpm_sums.nbytes) / 1e9,
        flush=True,
    )
    expression = lance.dataset(f"{atlas_root}/{SOURCE}/expression.lance")
    fragments = list(expression.get_fragments())
    records = 0
    for fi, fragment in enumerate(fragments):
        for attempt in range(12):
            try:
                table = fragment.to_table(columns=["cell_integer_id", "gene_integer_id", "value"])
                break
            except OSError:
                if attempt == 11:
                    raise
                print("fragment retry", fi, attempt, flush=True)
                time.sleep(min(5 * (attempt + 1), 60))
        ci = table.column("cell_integer_id").to_numpy().astype(np.int64)
        valid_id = ci <= max_id
        selected = np.zeros(len(ci), dtype=bool)
        selected[valid_id] = lookup[ci[valid_id]] >= 0
        if not selected.any():
            continue
        records += int(selected.sum())
        selected_ci = ci[selected]
        mapped = gene_map[table.column("gene_integer_id").to_numpy()[selected].astype(np.int64)]
        values = table.column("value").to_numpy()[selected].astype(np.float64)
        valid_gene = mapped >= 0
        rows = lookup[selected_ci[valid_gene]]
        columns = mapped[valid_gene]
        counts = values[valid_gene]
        np.add.at(count_sums, (rows, columns), counts)
        # Source rates use the source's full measured transcriptome library.
        np.add.at(
            cpm_sums,
            (rows, columns),
            1e6 * counts / np.maximum(total_lookup[selected_ci[valid_gene]], 1),
        )
        if fi % 50 == 0:
            print(
                "fragments",
                fi + 1,
                "/",
                len(fragments),
                "records",
                records,
                "seconds",
                round(time.time() - start_time),
                flush=True,
            )
    coverage = records / int(cells.n_genes_by_counts.sum())
    if coverage < 0.99:
        raise ValueError(f"Incomplete expression scan: {coverage}")
    control_by_sample = {sample: i for i, (sample, label) in enumerate(keys) if label == CONTROL}
    control_rows = np.asarray(list(control_by_sample.values()))
    global_counts = count_sums[control_rows].sum(axis=0)
    global_probability = global_counts / global_counts.sum()
    global_mean = cpm_sums[control_rows].sum(axis=0) / group_n[control_rows].sum()
    shape = (len(targets), len(genes))
    target_counts = np.zeros(shape)
    target_cpm = np.zeros(shape)
    matched_probability = np.zeros(shape)
    matched_cpm = np.zeros(shape)
    target_n = np.zeros(len(targets), dtype=np.int32)
    for ti, target in enumerate(targets):
        rows = [i for i, (_, label) in enumerate(keys) if label == target]
        if not rows:
            continue
        target_n[ti] = group_n[rows].sum()
        target_counts[ti] = count_sums[rows].sum(axis=0)
        target_cpm[ti] = cpm_sums[rows].sum(axis=0) / target_n[ti]
        all_library = target_counts[ti].sum()
        for row in rows:
            ci = control_by_sample.get(keys[row][0])
            if ci is None:
                probability, mean = global_probability, global_mean
            else:
                probability = count_sums[ci] / max(count_sums[ci].sum(), 1)
                mean = cpm_sums[ci] / group_n[ci]
            matched_probability[ti] += (count_sums[row].sum() / max(all_library, 1)) * probability
            matched_cpm[ti] += (group_n[row] / target_n[ti]) * mean
    np.savez_compressed(
        out / f"{SOURCE}_full_statistics.npz",
        source=np.asarray(SOURCE),
        targets=np.asarray(targets),
        genes=np.asarray(genes),
        n_cells=target_n,
        target_count_sums=target_counts.astype(np.float32),
        target_mean_cpm=target_cpm.astype(np.float32),
        matched_control_probability=matched_probability.astype(np.float32),
        matched_control_mean_cpm=matched_cpm.astype(np.float32),
        global_control_probability=global_probability.astype(np.float32),
        global_control_mean_cpm=global_mean.astype(np.float32),
        measured_genes=np.isin(np.arange(len(genes)), gene_map[gene_map >= 0]),
    )
    cells.to_parquet(out / f"{SOURCE}_selected_cells.parquet", index=False)
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "source": SOURCE,
                "url": f"{atlas_root}/{SOURCE}",
                "target_count": len(targets),
                "official_targets": len(official_targets),
                "additional_public_H1_targets": len(targets) - len(official_targets),
                "genes": len(genes),
                "source_library_normalization": "full source measured gene axis for per-cell CPM",
                "matched_control": "mix controls before taking logs; raw-library weights for pooled, cell weights for CPM",
                "selected_cells": len(cells),
                "coverage": coverage,
                "seconds": time.time() - start_time,
                "model_note": "No row normalization, low-rank reconstruction, sign gate, or panel-gene zeroing",
            },
            indent=2,
        )
    )
    print("EXTRACTION COMPLETE", SOURCE, flush=True)


if __name__ == "__main__":
    main()
