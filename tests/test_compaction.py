import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


def test_compaction_preserves_every_cell_and_its_metadata(tmp_path):
    pytest.importorskip("numba")
    rng = np.random.default_rng(3)
    index = [f"{c}_{t}_{i}" for c in "ABC" for t in ["t1", "t2"] for i in range(8)]
    obs = pd.DataFrame(
        {
            "context": np.repeat(list("ABC"), 16),
            "target_gene": np.tile(np.repeat(["t1", "t2"], 8), 3),
        },
        index=index,
    )
    raw = rng.integers(0, 20, (48, 12)).astype(np.float32)
    a = ad.AnnData(
        sparse.csr_matrix(raw), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(12)])
    )
    original, compact = tmp_path / "raw.h5ad", tmp_path / "compact.h5ad"
    a.write_h5ad(original)
    script = Path(__file__).resolve().parents[1] / "scripts/compact_prediction.py"
    subprocess.run(
        [sys.executable, str(script), str(original), str(compact), "--cells-per-target", "8"],
        check=True,
        capture_output=True,
        text=True,
    )
    b = ad.read_h5ad(compact)
    b = b[a.obs_names]
    np.testing.assert_array_equal(a.X.toarray(), b.X.toarray())
    np.testing.assert_array_equal(a.obs.columns, b.obs.columns)
    np.testing.assert_array_equal(a.obs.astype(str).to_numpy(), b.obs.astype(str).to_numpy())
    assert not (tmp_path / "compact.partial.h5ad").exists()
