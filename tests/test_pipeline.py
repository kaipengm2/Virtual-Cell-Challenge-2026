import json
import anndata as ad
import h5py
import numpy as np
import pytest

from vcc_atlas.demo import make_demo
from vcc_atlas.pipeline import load_config, predict


def test_end_to_end_counts_schema_and_reproducibility(tmp_path):
    config, data = make_demo(tmp_path / "example")
    output = tmp_path / "first"
    report = predict(config, data, output)
    assert report["status"] == "complete" and report["shape"] == [144, 24]
    assert not (output / "prediction.partial.h5ad").exists()
    a = ad.read_h5ad(output / "prediction.h5ad")
    assert a.obs_names.is_unique and a.var_names.is_unique
    assert a.obs.groupby(["context", "target_gene"], observed=True).size().eq(16).all()
    assert np.isfinite(a.X.data).all() and (a.X.data >= 0).all()
    np.testing.assert_array_equal(a.X.data, np.floor(a.X.data))
    with h5py.File(output / "prediction.h5ad") as f:
        assert f["X/indptr"].dtype == np.dtype("int64")
    predict(config, data, tmp_path / "second")
    b = ad.read_h5ad(tmp_path / "second/prediction.h5ad")
    assert (a.X != b.X).nnz == 0
    with pytest.raises(FileExistsError):
        predict(config, data, output)


def test_wrong_gene_order_fails_without_completed_file(tmp_path):
    config, data = make_demo(tmp_path / "example")
    a = ad.read_h5ad(data / "context_A.h5ad")
    a[:, ::-1].write_h5ad(data / "context_A.h5ad")
    with pytest.raises(ValueError, match="gene order"):
        predict(config, data, tmp_path / "output")
    assert not (tmp_path / "output/run.json").exists()
    assert not (tmp_path / "output/prediction.h5ad").exists()


@pytest.mark.parametrize(
    "key,value", [("pool_k", 0), ("seed", -1), ("extra", 3), ("contexts", ["A", "A"])]
)
def test_invalid_config(tmp_path, key, value):
    config, _ = make_demo(tmp_path / "example")
    spec = json.loads(config.read_text())
    spec[key] = value
    config.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        load_config(config)
