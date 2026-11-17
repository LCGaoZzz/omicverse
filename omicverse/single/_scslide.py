"""Optional scSLIDE-py adapter for sample-level cohort analysis.

No numerical implementation is vendored here. The backend is loaded only when
called. The initial integration deliberately exposes the descriptive v6 path,
not the experimental post-selection inference extensions.
"""
from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from .._registry import register_function

_BACKEND_VERSION = "0.3.3"
_OPTIONS = frozenset({
    "subject_id_sep", "dims_embedding", "ncells_per_group", "ncells_landmark",
    "ncomp_pls", "pls_function", "k_nn", "knn_range", "nn_method",
    "fix_wnn_weights", "min_lmc_count", "recompute_residuals_after_filter",
    "scale_max_sample", "ncomp_diffusion", "fit_curve", "de_celltype",
    "de_covariates", "de_inference_mode", "n_jobs", "seed", "reference_profile",
})


def _load_backend():
    try:
        backend = importlib.import_module("scslide")
    except ModuleNotFoundError as exc:
        if exc.name != "scslide":
            raise
        raise ImportError(
            "ov.single.scslide requires scslide-py 0.3.3 with its [anndata] "
            "extra. Install the SHA256-verified wheel from "
            "https://github.com/LCGaoZzz/scSLIDE-py; see the scSLIDE tutorial. "
            "This dependency is not installed automatically."
        ) from exc
    if getattr(backend, "__version__", None) != _BACKEND_VERSION:
        raise ImportError(
            "This initial adapter supports scslide-py 0.3.3 only; "
            f"found {getattr(backend, '__version__', 'unknown')!r}. "
            "Version alone does not authenticate an audited artifact: "
            "verify its SHA256 during installation."
        )
    return backend


def _matrix_values(matrix, name, shape, *, counts=False):
    import numpy as np
    from scipy import sparse

    if not isinstance(matrix, np.ndarray) and not sparse.issparse(matrix):
        raise TypeError(f"{name} must be an in-memory NumPy or SciPy sparse matrix")
    if matrix.shape != shape:
        raise ValueError(f"{name} has shape {matrix.shape}; expected {shape}")
    # Inspect stored sparse values, never a dense cells-by-genes conversion.
    values = matrix.tocsr(copy=False).data if sparse.issparse(matrix) else matrix
    if values.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numeric values")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must contain finite, non-negative values")
    if counts and not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name} must contain integer-valued raw counts")


def _validate_input(adata, sample_key, response_key, subject_key, celltype_key,
                    embedding_key, copy, overwrite, options):
    import numpy as np
    import pandas as pd

    if any(not isinstance(key, str) or not key for key in (sample_key, response_key)):
        raise ValueError("sample_key and response_key must be non-empty column names")
    for attr in ("obs", "var", "layers", "obsm", "uns", "obs_names", "var_names"):
        if not hasattr(adata, attr):
            raise TypeError("adata must be an AnnData-like cell-by-gene object")
    if getattr(adata, "isbacked", False):
        raise ValueError("The initial OV adapter supports in-memory AnnData only")
    if getattr(adata, "is_view", False) and not copy:
        raise ValueError("Use copy=True for an AnnData view")
    for name, ids in (("obs_names", adata.obs_names), ("var_names", adata.var_names)):
        ix = pd.Index(ids)
        if ix.hasnans or not ix.is_unique or not ix.astype(str).is_unique:
            raise ValueError(f"{name} must be non-missing and unique, including after string conversion")
    if len(adata.obs) == 0 or len(adata.var) == 0:
        raise ValueError("adata must contain cells and genes")
    for key in (sample_key, response_key, subject_key, celltype_key):
        if key is None:
            continue
        if key not in adata.obs or adata.obs[key].isna().any():
            raise ValueError(f"obs[{key!r}] is missing or contains missing values")
    samples = adata.obs[sample_key]
    if not samples.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("Sample identities must be non-empty strings")
    if samples.nunique() < 3:
        raise ValueError("At least three biological samples are required")
    for key in (response_key, subject_key):
        if key is not None and (adata.obs.groupby(sample_key, observed=True)[key].nunique() != 1).any():
            raise ValueError(f"obs[{key!r}] must be constant within each sample")
    if adata.obs[response_key].nunique() < 2:
        raise ValueError("response_key must contain at least two distinct values")
    if subject_key is not None and options.get("subject_id_sep") is not None:
        raise ValueError("Choose subject_key or subject_id_sep, not both")
    if "data" not in adata.layers:
        raise ValueError("layers['data'] must contain log-normalized, unscaled expression; no X fallback")
    shape = (len(adata.obs), len(adata.var))
    _matrix_values(adata.layers["data"], "layers['data']", shape)
    hv = adata.var.get("highly_variable")
    if hv is None or not pd.api.types.is_bool_dtype(hv) or hv.isna().any() or not hv.any():
        raise ValueError("var['highly_variable'] must be a non-missing Boolean mask with selected genes")
    if embedding_key is not None:
        if embedding_key not in adata.obsm:
            raise ValueError(f"obsm[{embedding_key!r}] is missing")
        emb = adata.obsm[embedding_key]
        if not isinstance(emb, np.ndarray) or emb.ndim != 2 or emb.shape[0] != shape[0]:
            raise ValueError("The complementary embedding must be a dense, cell-aligned 2D NumPy array")
        if not np.isfinite(emb).all():
            raise ValueError("The complementary embedding must contain finite values")
        dims = options.get("dims_embedding")
        if dims is not None:
            dims = list(dims)
            if not dims or len(set(dims)) != len(dims) or any(
                isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 0 or d >= emb.shape[1]
                for d in dims
            ):
                raise ValueError("dims_embedding must contain unique valid zero-based column indices")
            options["dims_embedding"] = dims
    elif options.get("dims_embedding") is not None:
        raise ValueError("dims_embedding requires embedding_key")
    if options.get("de_celltype") is not None:
        if celltype_key is None:
            raise ValueError("de_celltype requires celltype_key")
        if str(options["de_celltype"]) not in set(adata.obs[celltype_key].astype(str)):
            raise ValueError("de_celltype is absent from the input")
        if "counts" not in adata.layers:
            raise ValueError("Descriptive DE requires integer-valued raw layers['counts']")
        _matrix_values(adata.layers["counts"], "layers['counts']", shape, counts=True)
        for cov in options.get("de_covariates", ("Age", "Sex", "log_count")):
            if cov == "log_count":  # generated by the backend
                continue
            if cov not in adata.obs or adata.obs[cov].isna().any():
                raise ValueError(f"DE covariate {cov!r} is missing or contains missing values")
            if (adata.obs.groupby(sample_key, observed=True)[cov].nunique() != 1).any():
                raise ValueError(f"DE covariate {cov!r} must be constant within each sample")
    existing = "scslide" in adata.uns or "omicverse_scslide" in adata.uns or any(
        key == "X_proj.pls" or key.startswith("scslide_") for key in adata.obsm
    )
    if existing and not overwrite:
        raise ValueError("Existing scSLIDE results found; pass overwrite=True to recompute")


@register_function(
    aliases=["scSLIDE", "scslide", "sample-level landmark trajectory", "样本级单细胞轨迹"],
    category="single",
    description="Sample-level landmark representations and descriptive trajectories through "
                "the optional independent scslide-py backend; not cell pseudotime or calibrated causal inference.",
    examples=["result = ov.single.scslide(adata, sample_key='donor', response_key='disease')"],
    related=["single.mrvi", "single.pseudobulk"],
)
def scslide(adata: Any, *, sample_key: str, response_key: str,
            subject_key: str | None = None, celltype_key: str | None = None,
            embedding_key: str | None = None, copy: bool = True,
            overwrite: bool = False, workflow_kwargs: Mapping[str, Any] | None = None) -> Any:
    """Run the descriptive scSLIDE workflow and return its native WorkflowResult.

    Arguments:
        adata: In-memory cell-by-gene AnnData. ``layers['data']`` must hold
            log-normalized, unscaled expression and ``var['highly_variable']``
            a Boolean selection mask. Raw ``layers['counts']`` is additionally
            required when descriptive DE is requested. This adapter performs
            no implicit normalization, gene filtering, or embedding calculation.
        sample_key: Non-empty string biological sample IDs in ``obs`` (>=3 samples).
        response_key: Supervision/phenotype column, constant within each sample.
        subject_key: Optional subject IDs, constant within each sample. Use for
            repeated draws. If omitted without ``subject_id_sep``, the backend
            treats sample IDs as subjects; independence is not established by that.
        celltype_key: Optional cell-type column, required for descriptive DE.
        embedding_key: Exact ``obsm`` key for a complementary embedding (e.g.
            ``X_pca``). None uses the backend's single-modal PLS-neighbor path.
        copy: Copy input before computation (default True). The working cell
            object is ``result.prepared.adata``. False mutates input and is not
            transactional if a later computation fails.
        overwrite: Permit recomputation over existing backend-owned output keys.
        workflow_kwargs: Native workflow options (e.g. ``ncells_per_group``,
            ``ncells_landmark``, ``ncomp_pls``, ``ncomp_diffusion``, ``dims_embedding``,
            ``de_celltype``, ``de_covariates``, ``seed``, ``n_jobs``). Initial scope:
            ``reference_profile='v6'``, ``nn_method='exact'`` or ``'annoy'``, and
            ``de_inference_mode='descriptive_r_compat'`` only. Defaults are the
            native defaults; no experiment is silently substituted.

    Returns:
        scslide.WorkflowResult: ``sample_adata`` has samples as rows and landmarks
        as variables, NOT cells/genes. ``diffusion``, ``principal_curve``,
        ``de_results`` and ``inference_summary`` retain native semantics. The
        working cell object may gain ``X_proj.pls`` and ``scslide_*`` obsm entries
        and ``uns['scslide']``. No sample trajectory is written as cell pseudotime.

    Notes:
        The initial backend version is 0.3.3; authenticate the installed artifact
        separately by SHA256. Trajectory orientation is arbitrary and must be
        anchored biologically before interpreting DE signs. Descriptive adjusted
        p-values do not establish post-selection FDR/FWER control. This function
        returns a structured result rather than an AnnData, so automatic
        ``ov.report`` tracking is not claimed by this initial integration.
    """
    if not isinstance(copy, bool) or not isinstance(overwrite, bool):
        raise TypeError("copy and overwrite must be Boolean")
    if workflow_kwargs is not None and not isinstance(workflow_kwargs, Mapping):
        raise TypeError("workflow_kwargs must be a mapping")
    options = dict(workflow_kwargs or {})
    unknown = set(options).difference(_OPTIONS)
    if unknown:
        raise TypeError(f"Unsupported workflow options: {sorted(unknown)}")
    if options.get("reference_profile", "v6") != "v6":
        raise ValueError("The initial OV integration supports reference_profile='v6' only")
    if options.get("de_inference_mode", "descriptive_r_compat") != "descriptive_r_compat":
        raise ValueError("The initial OV integration exposes descriptive_r_compat only")
    if options.get("nn_method", "exact") not in ("exact", "annoy"):
        raise ValueError("Use nn_method='exact' or the optional 'annoy' backend")
    _validate_input(adata, sample_key, response_key, subject_key, celltype_key,
                    embedding_key, copy, overwrite, options)
    backend = _load_backend()
    working = adata.copy() if copy else adata
    result = backend.run_scslide_workflow(
        working, sample_col=sample_key, response_col=response_key,
        subject_col=subject_key, celltype_col=celltype_key,
        embedding_key=embedding_key, **options,
    )
    ids = result.sample_adata.obs_names.astype(str)
    if len(ids) == 0 or not ids.is_unique or not set(ids).issubset(set(adata.obs[sample_key])):
        raise RuntimeError("Backend returned invalid or misaligned sample identities")
    result.sample_adata.uns["omicverse_scslide"] = {
        "function": "ov.single.scslide", "backend_version": backend.__version__,
        "observation_unit": "sample", "variable_unit": "landmark",
        "input_copied": copy, "de_inference_mode": "descriptive_r_compat",
        "artifact_digest_verified_by_adapter": False,
    }
    return result


__all__ = ["scslide"]
