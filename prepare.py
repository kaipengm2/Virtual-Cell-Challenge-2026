"""Download public inputs and prepare the five v38 source statistics."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile
import anndata as ad
from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ASSETS = json.loads(Path(__file__).with_name("sources.json").read_text())
ATLAS = "hf://datasets/slaf-project/X-Atlas-Orion/data"
CONTROL = "Non-Targeting"


def download(name, directory):
    spec = ASSETS[name]
    path = directory / spec["filename"]
    directory.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pending = path.with_name(path.name + ".partial")
        remaining = spec["bytes"] - (pending.stat().st_size if pending.exists() else 0)
        if shutil.disk_usage(directory).free < remaining + 1024**3:
            raise ValueError("Insufficient disk space for " + name)
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "6",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "-C",
                "-",
                "-o",
                str(pending),
                spec["url"],
            ],
            check=True,
        )
    else:
        pending = path
    sha = hashlib.sha256()
    with pending.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            sha.update(block)
    if pending.stat().st_size != spec["bytes"] or sha.hexdigest() != spec["sha256"]:
        raise ValueError("Source checksum mismatch: " + name)
    if pending != path:
        pending.rename(path)
    return path


def controls(data, raw):
    names = [
        "gene_names.csv",
        "pert_counts.csv",
        "context_A.h5ad",
        "context_B.h5ad",
        "context_C.h5ad",
    ]
    if all((data / name).is_file() for name in names):
        return
    archive = raw / "controls.zip"
    subprocess.run(["vcc", "datasets", "download", "controls", "-o", str(archive)], check=True)
    with zipfile.ZipFile(archive) as zipped:
        for name in names:
            members = [m for m in zipped.namelist() if Path(m).name == name]
            if len(members) != 1:
                raise ValueError("Unexpected controls archive: " + name)
            pending = data / (name + ".partial")
            with zipped.open(members[0]) as src, pending.open("wb") as dest:
                shutil.copyfileobj(src, dest)
            pending.rename(data / name)


def source_targets(data, raw):
    official = pd.read_csv(data / "pert_counts.csv").target_gene.astype(str).tolist()
    h1 = pd.read_csv(download("h1-targets", raw)).target_gene.astype(str).tolist()
    return list(dict.fromkeys(official + [t for t in h1 if t != "non-targeting"]))


def prepare_k562(data, raw_dir):
    out = data
    path = download("k562", raw_dir)
    targets = source_targets(data, raw_dir)
    a = ad.read_h5ad(path, backed="r")
    if not {"gene", "gem_group"}.issubset(a.obs.columns):
        raise ValueError("Unexpected label/batch schema")
    labels = a.obs.gene.astype(str).to_numpy()
    batch_codes, batches = pd.factorize(a.obs.gem_group.astype(str))
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
    for left in range(0, len(rows), 1024):
        selected = rows[left : left + 1024]
        x = sparse.csr_matrix(a.X[selected]).astype(np.float64)
        if (
            not np.isfinite(x.data).all()
            or (x.data < 0).any()
            or (not np.equal(x.data, np.floor(x.data)).all())
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
        cpm_sums += (assignment @ x.multiply((1000000.0 / depth)[:, None])).toarray()
        perturb = group < n_targets
        np.add.at(target_batch_n, (group[perturb], batch_codes[selected][perturb]), 1)
        np.add.at(
            target_batch_library, (group[perturb], batch_codes[selected][perturb]), depth[perturb]
        )
    np.testing.assert_allclose(count_sums.sum(), source_library, rtol=1e-12)
    valid = group_n > 0
    np.testing.assert_allclose(cpm_sums[valid].sum(axis=1) / group_n[valid], 1000000.0, rtol=1e-10)
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
    a.file.close()


def prepare_cd4(data, raw_dir):
    out = data
    raw = download("cd4", raw_dir)
    targets = source_targets(data, raw_dir)
    with h5py.File(raw, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        genes = (
            var.gene_name.astype(str).to_numpy(dtype=str)
            if "gene_name" in var
            else var.index.astype(str).to_numpy(dtype=str)
        )
        if len(set(genes)) != len(genes):
            raise ValueError("Duplicate CD4 source symbols need explicit resolution")
        selected = np.flatnonzero(obs.target_contrast_gene_name.astype(str).isin(targets))
        chosen = obs.iloc[selected].copy()
        layers = {}
        for name in ["log_fc", "adj_p_value", "lfcSE"]:
            ds = f["layers"][name]
            if not isinstance(ds, h5py.Dataset) or ds.ndim != 2:
                raise ValueError("Expected publisher dense DE layer")
            layers[name] = np.asarray(ds[selected, :], dtype=np.float32)
    conditions = sorted(chosen.culture_condition.astype(str).unique())
    shape = (len(conditions), len(targets), len(genes))
    effects = np.zeros(shape, np.float32)
    q = np.ones(shape, np.float32)
    se = np.zeros(shape, np.float32)
    available = np.zeros(shape[:2], bool)
    quality = np.zeros(shape[:2], bool)
    n_cells = np.zeros(shape[:2], np.int64)
    rows_per_result = np.zeros(shape[:2], np.int32)
    for c, condition in enumerate(conditions):
        for t, target in enumerate(targets):
            rows = np.flatnonzero(
                (chosen.culture_condition.astype(str).to_numpy() == condition)
                & (chosen.target_contrast_gene_name.astype(str).to_numpy() == target)
            )
            if not len(rows):
                continue
            metadata = chosen.iloc[rows]
            values = layers["log_fc"][rows]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite publisher log2FC values")
            effects[c, t] = values.mean(axis=0)
            q[c, t] = layers["adj_p_value"][rows].max(axis=0)
            se[c, t] = layers["lfcSE"][rows].mean(axis=0)
            available[c, t] = True
            rows_per_result[c, t] = len(rows)
            n_cells[c, t] = int(metadata.n_cells_target.min())
            quality[c, t] = bool(
                (
                    (metadata.n_guides >= 2)
                    & ~metadata.single_guide_estimate.astype(bool)
                    & metadata.ontarget_significant.astype(bool)
                    & ~metadata.distal_offtarget_flag.astype(bool)
                    & ~metadata.low_target_gex.astype(bool)
                ).all()
            )
    np.savez_compressed(
        out / "CD4_DE_statistics.npz",
        targets=np.asarray(targets),
        genes=genes,
        conditions=np.asarray(conditions),
        log2fc=effects,
        adjusted_p=q,
        lfcSE=se,
        available=available,
        quality_pass=quality,
        n_cells=n_cells,
        rows_per_result=rows_per_result,
    )


def prepare_xatlas(source, data, raw, atlas_root):
    import lance
    from slaf import SLAFArray

    out = data
    official_targets = pd.read_csv(data / "pert_counts.csv")["target_gene"].astype(str).tolist()
    official_genes = pd.read_csv(data / "gene_names.csv")["gene_name"].astype(str).tolist()
    h1_counts = pd.read_csv(download("h1-targets", raw))
    h1_genes = pd.read_csv(download("h1-genes", raw))
    h1_targets = h1_counts["target_gene"].astype(str).tolist()
    gene_column = "gene_name" if "gene_name" in h1_genes else h1_genes.columns[0]
    genes = list(dict.fromkeys(official_genes + h1_genes[gene_column].astype(str).tolist()))
    targets = list(
        dict.fromkeys(official_targets + [t for t in h1_targets if t != "non-targeting"])
    )
    atlas = SLAFArray(f"{atlas_root}/{source}")
    selected_labels = ", ".join(
        ("'" + label.replace("'", "''") + "'" for label in targets + [CONTROL])
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
    expression = lance.dataset(f"{atlas_root}/{source}/expression.lance")
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
        np.add.at(
            cpm_sums,
            (rows, columns),
            1000000.0 * counts / np.maximum(total_lookup[selected_ci[valid_gene]], 1),
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
                probability, mean = (global_probability, global_mean)
            else:
                probability = count_sums[ci] / max(count_sums[ci].sum(), 1)
                mean = cpm_sums[ci] / group_n[ci]
            matched_probability[ti] += count_sums[row].sum() / max(all_library, 1) * probability
            matched_cpm[ti] += group_n[row] / target_n[ti] * mean
    np.savez_compressed(
        out / f"{source}_full_statistics.npz",
        source=np.asarray(source),
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


def summarize(path):
    data = ad.read_h5ad(path, backed="r")
    try:
        labels = data.obs.target_gene.astype(str).to_numpy()
        groups = np.asarray(sorted(set(labels)), dtype=str)
        ids = pd.Categorical(labels, categories=groups).codes
        n = np.bincount(ids, minlength=len(groups))
        genes = data.var_names.astype(str).to_numpy(dtype=str)
        counts = np.zeros((len(groups), len(genes)), dtype=np.float64)
        cpm = np.zeros_like(counts)
        for left in range(0, len(labels), 4096):
            right = min(left + 4096, len(labels))
            x = sparse.csr_matrix(data.X[left:right], dtype=np.float64)
            lib = np.asarray(x.sum(axis=1)).ravel()
            if (
                (lib <= 0).any()
                or not np.isfinite(x.data).all()
                or (x.data < 0).any()
                or (x.data != np.floor(x.data)).any()
            ):
                raise ValueError("Invalid H1 raw counts")
            membership = sparse.csr_matrix(
                (np.ones(right - left), (ids[left:right], np.arange(right - left))),
                shape=(len(groups), right - left),
            )
            counts += (membership @ x).toarray()
            cpm += (membership @ x.multiply(1000000.0 / lib[:, None])).toarray()
            if left % (4096 * 8) == 0:
                print(f"{Path(path).name}: {right}/{len(labels)} cells", flush=True)
        return dict(
            targets=groups, genes=genes, n_cells=n, count_sums=counts, mean_cpm=cpm / n[:, None]
        )
    finally:
        data.file.close()


def assemble(summaries):
    targets, counts, means, n, cp, cm, splits = ([], [], [], [], [], [], [])
    genes = summaries[0]["genes"]
    for label, d in zip(["train", "validation", "test"], summaries, strict=True):
        if not np.array_equal(genes, d["genes"]):
            raise ValueError("H1 gene axes differ")
        controls = np.flatnonzero(d["targets"] == "non-targeting")
        if len(controls) != 1:
            raise ValueError("Expected one control group per H1 split")
        c = int(controls[0])
        ctrl = d["count_sums"][c] / d["count_sums"][c].sum()
        mean = d["mean_cpm"][c]
        if label == "train":
            global_cp, global_cm = (ctrl.copy(), mean.copy())
        for i, target in enumerate(d["targets"]):
            if target == "non-targeting":
                continue
            if str(target) in targets:
                raise ValueError("Overlapping H1 treated targets across splits")
            targets.append(str(target))
            counts.append(d["count_sums"][i].copy())
            means.append(d["mean_cpm"][i].copy())
            n.append(int(d["n_cells"][i]))
            cp.append(ctrl.copy())
            cm.append(mean.copy())
            splits.append(label)
    return dict(
        source=np.asarray("H1_2025_public"),
        targets=np.asarray(targets),
        genes=genes,
        n_cells=np.asarray(n),
        target_count_sums=np.asarray(counts, dtype=np.float32),
        target_mean_cpm=np.asarray(means, dtype=np.float32),
        matched_control_probability=np.asarray(cp, dtype=np.float32),
        matched_control_mean_cpm=np.asarray(cm, dtype=np.float32),
        global_control_probability=global_cp.astype(np.float32),
        global_control_mean_cpm=global_cm.astype(np.float32),
        measured_genes=np.ones(len(genes), dtype=bool),
        public_2025_split=np.asarray(splits),
    )


def prepare_h1(data, raw):
    summaries = [
        summarize(download("h1-" + split, raw)) for split in ["train", "validation", "test"]
    ]
    result = assemble(summaries)
    if len(result["targets"]) != 300:
        raise ValueError("Expected 300 public H1 targets")
    np.savez_compressed(data / "H1_2025_full_statistics.npz", **result)


def pairs(table, targets, genes):
    unique = table[~table.gene.duplicated(keep=False)].set_index("gene")
    relevant = unique.loc[unique.index.intersection(genes)]
    rows = []
    for target in targets:
        if target not in unique.index:
            continue
        a = unique.loc[target]
        near = relevant[
            (relevant.chromosome == a.chromosome) & ((relevant.tss - a.tss).abs() <= 5000)
        ]
        for name, b in near.iterrows():
            if name == target:
                continue
            divergent = a.strand != b.strand and (
                a.strand == "+" and a.tss > b.tss or (a.strand == "-" and a.tss < b.tss)
            )
            rows.append(
                dict(
                    target=target,
                    neighbor=name,
                    distance=abs(int(a.tss) - int(b.tss)),
                    chromosome=a.chromosome,
                    target_tss=int(a.tss),
                    neighbor_tss=int(b.tss),
                    target_strand=a.strand,
                    neighbor_strand=b.strand,
                    divergent=divergent,
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "target",
            "neighbor",
            "distance",
            "chromosome",
            "target_tss",
            "neighbor_tss",
            "target_strand",
            "neighbor_strand",
            "divergent",
        ],
    )


def prepare_promoters(data, raw):
    gtf = download("gencode47", raw)
    rows = []
    with gzip.open(gtf, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[2] != "gene":
                continue
            attrs = dict(re.findall('(\\w+) "([^"]*)"', fields[8]))
            if "gene_name" not in attrs:
                continue
            rows.append(
                dict(
                    gene=attrs["gene_name"],
                    chromosome=fields[0],
                    strand=fields[6],
                    tss=int(fields[3] if fields[6] == "+" else fields[4]),
                )
            )
    result = pairs(
        pd.DataFrame(rows),
        pd.read_csv(data / "pert_counts.csv").target_gene.astype(str).to_numpy(),
        pd.read_csv(data / "gene_names.csv").gene_name.astype(str).to_numpy(),
    )
    result.to_csv(data / "official_pairs.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument(
        "--source", choices=["all", "k562", "hct", "hek", "h1", "cd4", "promoters"], default="all"
    )
    parser.add_argument("--atlas-root", default=ATLAS)
    args = parser.parse_args()
    data, raw = args.data_dir, args.raw_dir
    data.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    controls(data, raw)
    jobs = {
        "k562": ("K562_GWPS_CPM_full_statistics.npz", lambda: prepare_k562(data, raw)),
        "hct": (
            "HCT116_full_statistics.npz",
            lambda: prepare_xatlas("HCT116", data, raw, args.atlas_root),
        ),
        "hek": (
            "HEK293T_full_statistics.npz",
            lambda: prepare_xatlas("HEK293T", data, raw, args.atlas_root),
        ),
        "h1": ("H1_2025_full_statistics.npz", lambda: prepare_h1(data, raw)),
        "cd4": ("CD4_DE_statistics.npz", lambda: prepare_cd4(data, raw)),
        "promoters": ("official_pairs.csv", lambda: prepare_promoters(data, raw)),
    }
    for name, (filename, run) in jobs.items():
        if args.source not in ["all", name] or (data / filename).exists():
            continue
        print("Preparing " + name, flush=True)
        run()
        print("Saved " + filename, flush=True)


if __name__ == "__main__":
    main()
