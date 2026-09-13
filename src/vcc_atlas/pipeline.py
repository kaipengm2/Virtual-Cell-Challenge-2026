"""Config-driven v38 inference. No training service or competition login required."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .cd4 import aligned_cd4, add_cd4_family
from .fusion import fuse_source_centered
from .io import CountWriter, axis_csv, digest, validate_statistics
from .moments import dual_moment_counts
from .promoter import apply_promoter_prior
from .transfer import desired_mean, load_xatlas, seed_for


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {
        "schema_version",
        "sources",
        "cd4_path",
        "cd4_weight",
        "promoter_pairs",
        "promoter_fraction",
        "contexts",
        "cells_per_target",
        "pool_k",
        "seed",
        "prior_counts",
        "common_subtract",
        "cpm_amplitude",
        "bulk_amplitude",
        "effect_clip",
        "expected_targets",
        "expected_genes",
    }
    if set(config) != required:
        raise ValueError(
            f"Config keys differ: missing={required - set(config)}, unknown={set(config) - required}"
        )
    if config["schema_version"] != 1:
        raise ValueError("Unsupported config schema")
    for name in ["cells_per_target", "pool_k", "expected_targets", "expected_genes"]:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32 - 3:
        raise ValueError("Invalid seed")
    for name in [
        "cd4_weight",
        "prior_counts",
        "common_subtract",
        "cpm_amplitude",
        "bulk_amplitude",
        "effect_clip",
    ]:
        if (
            not isinstance(config[name], (int, float))
            or not np.isfinite(config[name])
            or config[name] < 0
        ):
            raise ValueError(f"Invalid {name}")
    if not 0 < config["promoter_fraction"] <= 1:
        raise ValueError("Invalid promoter fraction")
    contexts = config["contexts"]
    if (
        not isinstance(contexts, list)
        or not contexts
        or len(set(contexts)) != len(contexts)
        or not set(contexts) <= set("ABC")
    ):
        raise ValueError("Contexts must be a unique ordered subset of A, B, C")
    if not config["sources"]:
        raise ValueError("At least one source is required")
    for source in config["sources"]:
        if set(source) != {"path", "weight"} or not isinstance(source["path"], str):
            raise ValueError("Each source needs path and weight")
        if (
            not isinstance(source["weight"], (int, float))
            or not np.isfinite(source["weight"])
            or source["weight"] < 0
        ):
            raise ValueError("Invalid source weight")
    if sum(s["weight"] for s in config["sources"]) <= 0:
        raise ValueError("At least one source weight must be positive")
    if config["cd4_path"] is None and config["cd4_weight"] != 0:
        raise ValueError("CD4 weight requires a CD4 file")
    return config


def control_template(path, genes, cells, pool_k, seed):
    controls = ad.read_h5ad(path)
    if not np.array_equal(controls.var_names, genes):
        raise ValueError("Control gene order differs from gene_names.csv")
    raw = sparse.csr_matrix(controls.X, dtype=np.float64)
    if (
        not np.isfinite(raw.data).all()
        or (raw.data < 0).any()
        or (raw.data != np.floor(raw.data)).any()
    ):
        raise ValueError("Controls must contain raw nonnegative integer counts")
    library = np.asarray(raw.sum(axis=1)).ravel()
    if (library <= 0).any() or len(library) < cells * pool_k:
        raise ValueError(
            "Need positive-depth controls and at least cells_per_target * pool_k donors"
        )
    mean = np.zeros(len(genes))
    # Preserve the historical reduction order for numerical reproducibility.
    for left in range(0, len(library), 256):
        block = raw[left : left + 256].toarray() / library[left : left + 256, None]
        mean += block.sum(axis=0)
    mean /= len(library)
    bulk = np.asarray(raw.sum(axis=0)).ravel()
    bulk /= bulk.sum()
    selected = np.random.default_rng(seed).choice(len(library), cells * pool_k, replace=False)
    selected = selected[np.argsort(library[selected], kind="stable")]
    template = raw[selected].toarray() / library[selected, None]
    template = template.reshape(cells, pool_k, len(genes)).mean(axis=1)
    depths = np.rint(library[selected].reshape(cells, pool_k).mean(axis=1)).astype(np.int64)
    if depths.min() < 1 or depths.max() > 1_000_000:
        raise ValueError("Invalid predicted library depths")
    return template, depths, mean, bulk


def predict(config_path, data_dir, output_dir):
    started = time.monotonic()
    config = load_config(config_path)
    root, out = Path(data_dir), Path(output_dir)
    targets = axis_csv(root / "pert_counts.csv", "target_gene")
    genes = axis_csv(root / "gene_names.csv", "gene_name")
    if (len(targets), len(genes)) != (config["expected_targets"], config["expected_genes"]):
        raise ValueError("Input panel dimensions differ from config")
    cells = config["cells_per_target"]
    panel = pd.read_csv(root / "pert_counts.csv")
    if "n_cells" in panel and not (pd.to_numeric(panel.n_cells, errors="raise") == cells).all():
        raise ValueError(
            "This implementation requires uniform panel n_cells matching cells_per_target"
        )
    paths = [root / source["path"] for source in config["sources"]]
    for path in paths:
        validate_statistics(path)
    sources = [load_xatlas(path, prior_counts=config["prior_counts"]) for path in paths]
    weights = [source["weight"] for source in config["sources"]]
    cd4 = cd4_available = None
    if config["cd4_path"]:
        cd4, cd4_available = aligned_cd4(
            root / config["cd4_path"],
            targets,
            genes,
            common_subtract=config["common_subtract"],
            center_scope="source",
        )
    input_paths = paths + [root / "gene_names.csv", root / "pert_counts.csv"]
    input_paths += [root / f"context_{c}.h5ad" for c in config["contexts"]]
    input_paths += [root / config[key] for key in ["cd4_path", "promoter_pairs"] if config[key]]
    # Roles are relative identifiers; avoid recording machine-specific absolute paths.
    inputs = {
        str(p.relative_to(root)) if p.is_relative_to(root) else p.name: digest(p)
        for p in input_paths
    }
    code = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    base = {}
    for space in ["log2fc", "bulk_delta"]:
        base[space] = fuse_source_centered(
            sources, weights, targets, genes, space=space, common_subtract=config["common_subtract"]
        )
    del sources
    # Require a new output directory; failed jobs cannot be mistaken for completed ones.
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    temporary = out / "prediction.partial.h5ad"
    effects, probabilities, coverage, bulks, reports = [], [], [], [], []
    with CountWriter(temporary, targets, genes, config["contexts"], cells) as writer:
        for ci, context in enumerate(config["contexts"]):
            template, depths, mean, bulk = control_template(
                root / f"context_{context}.h5ad",
                genes,
                cells,
                config["pool_k"],
                config["seed"] + ci,
            )
            desired, changes = {}, {}
            for space, profile, amplitude in [
                ("log2fc", mean, config["cpm_amplitude"]),
                ("bulk_delta", bulk, config["bulk_amplitude"]),
            ]:
                effect, covered = base[space]
                if cd4 is not None:
                    effect, covered = add_cd4_family(
                        effect,
                        covered,
                        cd4,
                        cd4_available,
                        config["cd4_weight"],
                        space=space,
                        control_probability=profile,
                    )
                probability = desired_mean(
                    profile, effect, space=space, amplitude=amplitude, clip=config["effect_clip"]
                )
                if config["promoter_pairs"]:
                    probability, changes[space] = apply_promoter_prior(
                        probability,
                        profile,
                        targets,
                        genes,
                        root / config["promoter_pairs"],
                        config["promoter_fraction"],
                    )
                desired[space] = probability
                if space == "log2fc":
                    effects.append(effect)
                    coverage.append(covered)
            probabilities.append(desired["log2fc"])
            bulks.append(desired["bulk_delta"])
            fits = []
            before = writer.nnz
            for ti, target in enumerate(targets):
                counts, fit = dual_moment_counts(
                    template,
                    desired["log2fc"][ti],
                    desired["bulk_delta"][ti],
                    depths=depths,
                    seed=seed_for(context + ":" + target, config["seed"]),
                )
                fit["cpm_l1_after_rounding"] = float(
                    np.abs((counts / depths[:, None]).mean(axis=0) - desired["log2fc"][ti]).sum()
                )
                writer.append(counts)
                fits.append({"target": str(target), **fit})
                if (ti + 1) % 25 == 0 or ti + 1 == len(targets):
                    print(
                        f"{context}: {ti + 1}/{len(targets)} targets; {time.monotonic() - started:.1f}s",
                        flush=True,
                    )
            (out / f"moment_fit_{context}.json").write_text(json.dumps(fits, indent=2) + "\n")
            (out / f"promoter_changes_{context}.json").write_text(
                json.dumps(changes, indent=2) + "\n"
            )
            reports.append(
                {
                    "context": context,
                    "nnz": writer.nnz - before,
                    "depth_min": int(depths.min()),
                    "depth_max": int(depths.max()),
                    "max_cpm_l1_after_rounding": max(f["cpm_l1_after_rounding"] for f in fits),
                }
            )
    axes = {"targets": targets, "genes": genes, "contexts": np.asarray(config["contexts"])}
    np.savez_compressed(
        out / "effects.npz",
        **axes,
        effects=np.asarray(effects),
        desired_probabilities=np.asarray(probabilities),
        source_weight=np.asarray(coverage),
    )
    np.savez_compressed(out / "requested_bulk.npz", **axes, probabilities=np.asarray(bulks))
    if code != {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}:
        raise RuntimeError("Source code changed during prediction; rerun with a fixed installation")
    path = out / "prediction.h5ad"
    temporary.rename(path)
    report = {
        "status": "complete",
        "shape": [writer.nobs, len(genes)],
        "nnz": writer.nnz,
        "contexts": reports,
        "seconds": time.monotonic() - started,
        "prediction_sha256": digest(path),
        "input_sha256": inputs,
        "code_sha256": code,
        "python": platform.python_version(),
        "platform": platform.system(),
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["vcc-atlas-transfer", "numpy", "scipy", "pandas", "anndata", "h5py"]
        },
    }
    (out / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
