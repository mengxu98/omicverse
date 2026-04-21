r"""Tissue-zone / cellular-compartment discovery via NMF.

After a spatial deconvolver (Cell2location, Tangram, SpatialDWLS, ...)
writes a per-spot cell-type abundance matrix, a common follow-up is to
ask *which groups of cell types co-locate?* That answer becomes
**tissue zones** (germinal centre, T-cell zone, stromal compartment …):
small, interpretable compartments defined by a handful of co-occurring
cell types.

The canonical Cell2location recipe uses a Bayesian `CoLocatedGroupsSklearnNMF`
(vendored at ``omicverse.external.space.cell2location.run_colocation``);
it is faithful but heavy (10 min+ on a single Visium slide, depends on
pyro). The function here gives you the **same tissue-zone output**
via a plain ``sklearn.decomposition.NMF`` on the cell-abundance matrix
— seconds instead of minutes, and no extra optional deps beyond
scikit-learn.

See `cell2location's tutorial
<https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_tutorial.html#Identifying-cellular-compartments-/-tissue-zones-using-matrix-factorisation-(NMF)>`_
for the biology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from .._registry import register_function


@dataclass
class TissueZones:
    """Result of :func:`nmf_tissue_zones`.

    Attributes
    ----------
    factor_loadings : pd.DataFrame, (n_cell_types, n_factors)
        Per-factor loading on every cell type. Row sums are **not**
        normalised — the loading is the raw ``components_`` output of
        sklearn NMF. Compare relative magnitudes across factors within
        the same cell type.
    spot_activations : pd.DataFrame, (n_spots, n_factors)
        Per-spot activation of each tissue zone, raw NMF scores. Also
        written to ``adata.obsm[obsm_added]`` as a numpy array.
    factor_top_cell_types : dict[str, list[str]]
        For each factor, the ``top_k`` cell types with the highest
        loadings — a quick handle on what each zone represents.
    factor_names : list[str]
        Human-readable factor labels (defaults to ``"zone_1"``, …).
    n_factors : int
    reconstruction_err : float
    """

    factor_loadings: pd.DataFrame
    spot_activations: pd.DataFrame
    factor_top_cell_types: dict
    factor_names: list
    n_factors: int
    reconstruction_err: float


@register_function(
    aliases=[
        "nmf_tissue_zones",
        "nmf_compartments",
        "cellular_compartments",
        "组织区域分解",
        "spatial_nmf_colocation",
    ],
    category="space",
    description=(
        "Factorise a spatial cell-abundance matrix into tissue zones / "
        "cellular compartments via scikit-learn NMF. Lightweight "
        "alternative to cell2location's Bayesian CoLocatedGroupsSklearnNMF "
        "with the same interpretation."
    ),
    examples=[
        "tz = ov.space.nmf_tissue_zones(adata, "
        "obsm_key='q05_cell_abundance_w_sf', n_factors=10)",
        "# spot activations in adata.obsm['X_tissue_zones']",
        "# factor loadings in tz.factor_loadings",
    ],
    related=[
        "space.Deconvolution",
        "space.CellMap",
        "utils.mde",
    ],
)
def nmf_tissue_zones(
    adata: AnnData,
    obsm_key: str = "q05_cell_abundance_w_sf",
    n_factors: int = 10,
    *,
    cell_type_names: Optional[Sequence[str]] = None,
    top_k: int = 5,
    obsm_added: str = "X_tissue_zones",
    factor_prefix: str = "zone",
    init: str = "nndsvd",
    max_iter: int = 500,
    tol: float = 1e-4,
    seed: int = 0,
) -> TissueZones:
    """Discover tissue zones via NMF on a per-spot cell-abundance matrix.

    Parameters
    ----------
    adata
        Spatial AnnData with a cell-abundance matrix in ``adata.obsm``.
    obsm_key
        Where to read the cell-abundance matrix from. Cell2location
        writes ``q05_cell_abundance_w_sf`` (q05 posterior estimate of
        absolute abundance × size factor) — the recommended input for
        downstream interpretation. Other deconvolvers may use
        different keys; pass whatever 2-D float array ``adata.obsm``
        entry you want to factorise.
    n_factors
        Number of tissue zones to extract. Cell2location's tutorial
        recommends trying several values (``n_fact = [5, 10, 15, 20]``)
        and picking the one that separates biology best. Start with
        around the number of cell types you expect to cluster together.
    cell_type_names
        Explicit names for the cell-type axis. When omitted we read
        ``adata.uns[f'{obsm_key}_names']`` (Cell2location writes this)
        or fall back to ``cell_type_0 ... cell_type_{C-1}``.
    top_k
        For each factor, report the ``top_k`` cell types with the
        highest loadings in ``result.factor_top_cell_types``. Default 5.
    obsm_added
        Where to write the per-spot factor activations. Default
        ``X_tissue_zones``. Also written back to
        ``adata.obsm[obsm_added]`` so downstream plotting (Scanpy
        ``sc.pl.spatial``, ``ov.pl.embedding``) can colour spots by
        zone directly.
    factor_prefix
        Prefix for factor names; the i-th factor is
        ``f'{factor_prefix}_{i+1}'``.
    init, max_iter, tol, seed
        Forwarded to ``sklearn.decomposition.NMF``. ``init='nndsvd'``
        gives deterministic initialisation; ``'random'`` + ``seed``
        gives multiple-restart behaviour if you loop manually.

    Returns
    -------
    TissueZones
        See the dataclass for attribute details.

    Notes
    -----
    - The abundance matrix **must be non-negative**. Most deconvolvers
      produce non-negative output; if your matrix has small negative
      values (numerical noise) they are clipped to 0 here.
    - This is a lightweight sibling of Cell2location's
      ``run_colocation`` (pyro-backed, Bayesian). The output is
      qualitatively equivalent — factor loadings and spot activations
      with the same interpretation — but the sklearn NMF runs in
      seconds and has no pyro/torch dependency.

    References
    ----------
    .. [1] Kleshchevnikov et al., *Nat. Biotechnol.* 2022 — Cell2location.
       Tissue-zone discovery via matrix factorisation of the per-spot
       cell-abundance matrix.
    """
    from sklearn.decomposition import NMF

    if obsm_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm has no key {obsm_key!r}. Available: "
            f"{list(adata.obsm.keys())[:8]}."
        )
    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"obsm[{obsm_key!r}] must be 2-D, got shape {X.shape}."
        )
    if np.any(~np.isfinite(X)):
        n_bad = int((~np.isfinite(X)).sum())
        raise ValueError(
            f"obsm[{obsm_key!r}] has {n_bad} non-finite entries; "
            f"NMF requires finite non-negative input."
        )
    # Clip small numerical-noise negatives; raise if anything is
    # meaningfully below zero.
    min_val = float(X.min())
    if min_val < -1e-8:
        raise ValueError(
            f"obsm[{obsm_key!r}] has negative values as low as "
            f"{min_val:.4g}. NMF requires non-negative input — check "
            f"your deconvolver output."
        )
    if min_val < 0:
        X = np.clip(X, 0.0, None)

    # Resolve cell-type names
    if cell_type_names is None:
        cand = adata.uns.get(f"{obsm_key}_names")
        if cand is not None and len(cand) == X.shape[1]:
            cell_type_names = list(cand)
        else:
            cand = adata.uns.get("mod", {}).get("factor_names")
            if cand is not None and len(cand) == X.shape[1]:
                cell_type_names = list(cand)
            else:
                cell_type_names = [f"cell_type_{i}"
                                   for i in range(X.shape[1])]
    else:
        cell_type_names = list(cell_type_names)
        if len(cell_type_names) != X.shape[1]:
            raise ValueError(
                f"len(cell_type_names)={len(cell_type_names)} but "
                f"obsm[{obsm_key!r}] has {X.shape[1]} cell types."
            )

    factor_names = [f"{factor_prefix}_{i + 1}" for i in range(n_factors)]

    model = NMF(
        n_components=n_factors,
        init=init,
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
    )
    W = model.fit_transform(X)        # (n_spots, n_factors)
    H = model.components_              # (n_factors, n_cell_types)

    spot_activations = pd.DataFrame(
        W, index=adata.obs_names.copy(), columns=factor_names,
    )
    factor_loadings = pd.DataFrame(
        H.T, index=cell_type_names, columns=factor_names,
    )
    # Top-k cell types per factor
    factor_top = {
        f: factor_loadings[f]
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
        for f in factor_names
    }

    adata.obsm[obsm_added] = W
    adata.uns.setdefault("tissue_zones", {})[obsm_added] = {
        "n_factors": int(n_factors),
        "source_obsm": obsm_key,
        "cell_type_names": cell_type_names,
        "factor_names": factor_names,
        "reconstruction_err": float(model.reconstruction_err_),
    }

    return TissueZones(
        factor_loadings=factor_loadings,
        spot_activations=spot_activations,
        factor_top_cell_types=factor_top,
        factor_names=factor_names,
        n_factors=int(n_factors),
        reconstruction_err=float(model.reconstruction_err_),
    )
