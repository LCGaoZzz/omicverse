"""Adapter contract tests; backend equivalence is an explicit optional test.

These tests load only the dispatcher with a registry stub, not the heavyweight
OmicVerse package. This is deliberately not a full ``import omicverse`` test.
"""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


@pytest.fixture
def adapter(monkeypatch):
    metadata = {}
    package = ModuleType("_scslide_adapter_test")
    package.__path__ = []
    single = ModuleType("_scslide_adapter_test.single")
    single.__path__ = []
    registry = ModuleType("_scslide_adapter_test._registry")

    def register_function(**kwargs):
        metadata.update(kwargs)
        return lambda fn: fn

    registry.register_function = register_function
    for module in (package, single, registry):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    source = Path(__file__).resolve().parents[2] / "omicverse/single/_scslide.py"
    spec = importlib.util.spec_from_file_location("_scslide_adapter_test.single._scslide", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.registration = metadata
    return mod


class MiniAnnData:
    """Only the adapter-facing protocol; not a replacement for backend AnnData."""
    def __init__(self):
        self.obs = pd.DataFrame({
            "sample": ["S1", "S1", "S2", "S2", "S3", "S3"],
            "disease": [0, 0, 1, 1, 2, 2],
            "subject": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "celltype": ["T", "B"] * 3,
            "Age": [40, 40, 50, 50, 60, 60],
            "Sex": ["F", "F", "M", "M", "F", "F"],
        }, index=[f"C{i}" for i in range(6)])
        self.var = pd.DataFrame({"highly_variable": [True, True, False]}, index=["G1", "G2", "G3"])
        self.layers = {"data": sparse.csr_matrix(np.ones((6, 3))), "counts": sparse.csr_matrix(np.ones((6, 3)))}
        self.obsm = {"X_pca": np.ones((6, 2))}
        self.uns = {}
        self.isbacked = False
        self.is_view = False

    @property
    def obs_names(self):
        return self.obs.index

    @property
    def var_names(self):
        return self.var.index

    def copy(self):
        return copy.deepcopy(self)


@pytest.fixture
def data():
    return MiniAnnData()


@pytest.fixture
def fake_backend(adapter, monkeypatch):
    calls = []

    def run(adata, **kwargs):
        calls.append((adata, kwargs))
        adata.uns["scslide"] = {"computed": True}
        result = SimpleNamespace(
            sample_adata=SimpleNamespace(obs_names=pd.Index(["S2", "S1", "S3"]), uns={}),
            prepared=SimpleNamespace(adata=adata),
        )
        return result

    backend = SimpleNamespace(__version__="0.3.3", run_scslide_workflow=run)
    monkeypatch.setattr(adapter, "_load_backend", lambda: backend)
    return backend, calls


def test_registry_metadata(adapter):
    assert adapter.registration["category"] == "single"
    assert "scSLIDE" in adapter.registration["aliases"]
    assert adapter.registration["examples"]


def test_copy_forwarding_and_sample_units(adapter, data, fake_backend):
    result = adapter.scslide(data, sample_key="sample", response_key="disease", subject_key="subject",
                             celltype_key="celltype", embedding_key="X_pca",
                             workflow_kwargs={"seed": 42, "n_jobs": 1, "dims_embedding": range(2)})
    working, options = fake_backend[1][0]
    assert working is not data and data.uns == {}
    assert options["sample_col"] == "sample" and options["subject_col"] == "subject"
    assert options["seed"] == 42 and options["dims_embedding"] == [0, 1]
    assert list(result.sample_adata.obs_names) == ["S2", "S1", "S3"]  # no positional relabeling
    meta = result.sample_adata.uns["omicverse_scslide"]
    assert meta["observation_unit"] == "sample" and not meta["artifact_digest_verified_by_adapter"]
    assert "traj_overall" not in data.obs


def test_copy_false_and_overwrite(adapter, data, fake_backend):
    data.uns["scslide"] = {"old": True}
    with pytest.raises(ValueError, match="overwrite"):
        adapter.scslide(data, sample_key="sample", response_key="disease")
    result = adapter.scslide(data, sample_key="sample", response_key="disease", copy=False, overwrite=True)
    assert result.prepared.adata is data
    assert data.uns["scslide"]["computed"]


@pytest.mark.parametrize("change, message", [
    (lambda a: a.layers.pop("data"), "no X fallback"),
    (lambda a: a.var.pop("highly_variable"), "Boolean mask"),
    (lambda a: a.var.__setitem__("highly_variable", [1, 1, 0]), "Boolean mask"),
    (lambda a: a.var.__setitem__("highly_variable", [False] * 3), "Boolean mask"),
    (lambda a: a.obs.__setitem__("sample", ["S1"] * 6), "three biological"),
    (lambda a: a.obs.__setitem__("sample", [1, 1, 2, 2, 3, 3]), "non-empty strings"),
    (lambda a: a.obs.__setitem__("disease", [0] * 6), "distinct values"),
    (lambda a: a.obs.loc.__setitem__(("C0", "disease"), 5), "constant within"),
    (lambda a: a.obs.loc.__setitem__(("C0", "sample"), None), "missing values"),
    (lambda a: setattr(a.obs, "index", ["x"] * 6), "obs_names"),
    (lambda a: setattr(a.var, "index", ["x"] * 3), "var_names"),
    (lambda a: setattr(a, "isbacked", True), "in-memory"),
    (lambda a: a.layers.__setitem__("data", np.full((6, 3), np.nan)), "finite"),
    (lambda a: a.layers.__setitem__("data", np.full((6, 3), -1.0)), "non-negative"),
])
def test_input_failures_are_preflight(adapter, data, fake_backend, change, message):
    change(data)
    with pytest.raises((ValueError, TypeError), match=message):
        adapter.scslide(data, sample_key="sample", response_key="disease")
    assert fake_backend[1] == []


@pytest.mark.parametrize("options,message", [
    ({"sample_col": "x"}, "Unsupported"),
    ({"typo": 2}, "Unsupported"),
    ({"reference_profile": "legacy"}, "v6"),
    ({"de_inference_mode": "confirmatory_oof_countsplit"}, "descriptive_r_compat"),
    ({"nn_method": "pynndescent"}, "exact"),
    ({"dims_embedding": [0]}, "requires embedding_key"),
    ({"de_celltype": "T"}, "requires celltype_key"),
])
def test_scope_guards(adapter, data, fake_backend, options, message):
    with pytest.raises((ValueError, TypeError), match=message):
        adapter.scslide(data, sample_key="sample", response_key="disease", workflow_kwargs=options)
    assert fake_backend[1] == []


@pytest.mark.parametrize("dims", [[-1], [2], [0, 0], [0.5], [True], []])
def test_invalid_dimensions_not_silently_truncated(adapter, data, fake_backend, dims):
    with pytest.raises(ValueError, match="dims_embedding"):
        adapter.scslide(data, sample_key="sample", response_key="disease", embedding_key="X_pca",
                        workflow_kwargs={"dims_embedding": dims})


def test_sparse_validation_never_densifies(adapter, data, fake_backend, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("adapter densified sparse matrix")
    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    adapter.scslide(data, sample_key="sample", response_key="disease")


def test_raw_counts_and_covariates_are_required_for_de(adapter, data, fake_backend):
    options = {"de_celltype": "T"}
    data.layers["counts"] = sparse.csr_matrix(np.full((6, 3), 0.5))
    with pytest.raises(ValueError, match="integer-valued"):
        adapter.scslide(data, sample_key="sample", response_key="disease", celltype_key="celltype", workflow_kwargs=options)
    data.layers["counts"] = sparse.csr_matrix(np.ones((6, 3)))
    data.obs.pop("Age")
    with pytest.raises(ValueError, match="covariate"):
        adapter.scslide(data, sample_key="sample", response_key="disease", celltype_key="celltype", workflow_kwargs=options)


def test_only_missing_backend_is_reworded(adapter, monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("missing", name="scslide")
    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="SHA256"):
        adapter._load_backend()
    def broken(name):
        raise ModuleNotFoundError("backend dependency failure", name="scipy")
    monkeypatch.setattr(adapter.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="backend dependency failure"):
        adapter._load_backend()


def test_version_guard(adapter, monkeypatch):
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: SimpleNamespace(__version__="0.3.2"))
    with pytest.raises(ImportError, match="0.3.3 only"):
        adapter._load_backend()


def test_bad_output_ids_rejected(adapter, data, fake_backend):
    fake_backend[0].run_scslide_workflow = lambda *a, **k: SimpleNamespace(
        sample_adata=SimpleNamespace(obs_names=pd.Index(["other"]), uns={}))
    with pytest.raises(RuntimeError, match="sample identities"):
        adapter.scslide(data, sample_key="sample", response_key="disease")


@pytest.mark.skipif(os.environ.get("SCSLIDE_RUN_BACKEND_TESTS") != "1", reason="requires explicitly installed audited scslide-py wheel")
def test_exact_backend_equivalence_and_h5ad(adapter, tmp_path):
    import anndata as ad
    import scslide as sl

    data = sl.build_synthetic_cohort_fixture(seed=20261116)
    options = dict(ncells_per_group=20, ncells_landmark=40, knn_range=40,
                   ncomp_pls=4, ncomp_diffusion=4, dims_embedding=list(range(8)),
                   nn_method="exact", n_jobs=1, seed=123)
    direct = sl.run_scslide_workflow(data.copy(), sample_col="Donor", response_col="Disease",
                                     subject_col="subject_id", celltype_col="cell_type",
                                     embedding_key="X_pca", **options)
    wrapped = adapter.scslide(data, sample_key="Donor", response_key="Disease",
                              subject_key="subject_id", celltype_key="cell_type",
                              embedding_key="X_pca", workflow_kwargs=options)
    pd.testing.assert_frame_equal(direct.sample_object.counts, wrapped.sample_object.counts)
    np.testing.assert_array_equal(direct.prepared.neighbors.nn_idx, wrapped.prepared.neighbors.nn_idx)
    np.testing.assert_allclose(direct.diffusion.embeddings, wrapped.diffusion.embeddings, atol=1e-10)
    np.testing.assert_allclose(direct.principal_curve.lambda_, wrapped.principal_curve.lambda_, atol=1e-10)
    assert direct.inference_summary == wrapped.inference_summary
    assert "scslide" not in data.uns
    dest = tmp_path / "samples.h5ad"
    wrapped.sample_adata.write_h5ad(dest)
    restored = ad.read_h5ad(dest)
    assert restored.obs_names.equals(wrapped.sample_adata.obs_names)
    assert restored.uns["omicverse_scslide"]["observation_unit"] == "sample"
