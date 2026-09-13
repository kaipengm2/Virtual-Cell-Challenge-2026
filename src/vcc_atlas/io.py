"""Portable input validation and bounded-memory H5AD output."""

from __future__ import annotations

import hashlib
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def axis_csv(path, column):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if column not in table:
        raise ValueError(f"{Path(path).name}: expected a {column!r} column")
    values = table[column].to_numpy(dtype=str)
    if not len(values) or any(not x.strip() for x in values) or len(set(values)) != len(values):
        raise ValueError(f"{column}: labels must be nonempty and unique")
    if column == "target_gene" and "non-targeting" in values:
        raise ValueError("Prediction panel must not contain controls")
    return values


def validate_statistics(path):
    with np.load(path, allow_pickle=False) as data:
        targets, genes = data["targets"], data["genes"]
        for name, values in [("targets", targets), ("genes", genes)]:
            if values.ndim != 1 or values.dtype.kind not in "US" or not len(values):
                raise ValueError(f"{name}: expected a nonempty string axis")
            if len(set(values)) != len(values):
                raise ValueError(f"{name}: duplicate labels")
        shape = (len(targets), len(genes))
        for name in [
            "target_count_sums",
            "target_mean_cpm",
            "matched_control_probability",
            "matched_control_mean_cpm",
        ]:
            x = data[name]
            if x.shape != shape or not np.isfinite(x).all() or (x < 0).any():
                raise ValueError(f"{Path(path).name}: invalid {name}")
        n = data["n_cells"]
        if (
            n.shape != (shape[0],)
            or not np.isfinite(n).all()
            or (n < 0).any()
            or (n != np.floor(n)).any()
        ):
            raise ValueError("Invalid source cell counts")
        ctrl = data["matched_control_probability"].sum(axis=1)
        if not np.allclose(ctrl[n >= 20], 1, atol=1e-5):
            raise ValueError("Covered source control probabilities must sum to one")
        if data["measured_genes"].shape != (shape[1],) or data["measured_genes"].dtype.kind != "b":
            raise ValueError("Invalid measured gene mask")


class CountWriter:
    """Append integer count blocks; retain int64 CSR pointers beyond 2^31 NNZ."""

    def __init__(self, path, targets, genes, contexts, cells):
        self.path = Path(path)
        self.rows = 0
        self.nnz = 0
        self.nobs = len(targets) * len(contexts) * cells
        self.ngenes = len(genes)
        obs = pd.DataFrame(
            {
                "target_gene": np.tile(np.repeat(targets, cells), len(contexts)),
                "context": np.repeat(contexts, len(targets) * cells),
            },
            index=[f"{c}_{t}_{i}" for c in contexts for t in targets for i in range(cells)],
        )
        ad.AnnData(
            sparse.csr_matrix((self.nobs, len(genes)), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=genes),
        ).write_h5ad(self.path)
        self.file = h5py.File(self.path, "r+")
        group = self.file["X"]
        for name in ["data", "indices", "indptr"]:
            del group[name]
        self.data = group.create_dataset(
            "data",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            chunks=(1048576,),
            compression="lzf",
            shuffle=True,
        )
        self.indices = group.create_dataset(
            "indices",
            shape=(0,),
            maxshape=(None,),
            dtype="int32",
            chunks=(1048576,),
            compression="lzf",
            shuffle=True,
        )
        self.pointers = group.create_dataset("indptr", shape=(self.nobs + 1,), dtype="int64")
        self.pointers[0] = 0

    def append(self, counts):
        if (
            counts.ndim != 2
            or counts.shape[1] != self.ngenes
            or self.rows + len(counts) > self.nobs
        ):
            raise ValueError("Unexpected count block shape")
        if (
            not np.isfinite(counts).all()
            or (counts < 0).any()
            or (counts != np.floor(counts)).any()
        ):
            raise ValueError("Expected finite nonnegative integer counts")
        if (counts.sum(axis=1, dtype=np.float64) > 1_000_000).any():
            raise ValueError("Cell depth exceeds 1,000,000")
        block = sparse.csr_matrix(counts.astype(np.uint32))
        end = self.nnz + block.nnz
        if end > 4_750_000_000:
            raise ValueError("Sparse element limit exceeded")
        self.data.resize((end,))
        self.indices.resize((end,))
        self.data[self.nnz : end] = block.data
        self.indices[self.nnz : end] = block.indices
        self.pointers[self.rows + 1 : self.rows + len(counts) + 1] = (
            block.indptr[1:].astype(np.int64) + self.nnz
        )
        self.rows += len(counts)
        self.nnz = end

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        complete = self.rows == self.nobs and int(self.pointers[-1]) == self.nnz
        self.file.close()
        if kind is None and not complete:
            raise ValueError("Incomplete prediction output")
