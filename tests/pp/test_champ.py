"""Tests for ``ov.pp.champ`` (CHAMP — Weir et al. 2017).

Verifies:
- Modularity coefficients (a, b) match the closed-form formula on a
  trivial graph that we can hand-check.
- The upper-hull selection picks a hull partition (never a dominated
  one) on synthetic three-blob data.
- ``adata.uns['champ']`` payload schema, temp-column cleanup,
  ``@tracked`` provenance hookup, missing-graph error.
"""
from __future__ import annotations

import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _three_blob_adata(n_per_blob: int = 100, n_genes: int = 50,
                       seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    centers = np.zeros((3, n_genes))
    centers[0, :n_genes // 3] = 5
    centers[1, n_genes // 3:2 * n_genes // 3] = 5
    centers[2, 2 * n_genes // 3:] = 5
    X = np.vstack([
        centers[0] + rng.normal(0, 0.5, size=(n_per_blob, n_genes)),
        centers[1] + rng.normal(0, 0.5, size=(n_per_blob, n_genes)),
        centers[2] + rng.normal(0, 0.5, size=(n_per_blob, n_genes)),
    ]).astype(np.float32)
    obs = pd.DataFrame({"true_label": np.repeat(["A", "B", "C"], n_per_blob)},
                        index=[f"c{i}" for i in range(3 * n_per_blob)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture(scope="module")
def adata_with_neighbors():
    os.environ.setdefault("OMICVERSE_DISABLE_LLM", "1")
    import scanpy as sc
    a = _three_blob_adata(n_per_blob=120, n_genes=60, seed=0)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
    return a


# ─────────────────────── modularity coefficients ──────────────────────────────


def test_modularity_coefficients_two_perfect_clusters():
    """For a graph with two disconnected fully-connected components and
    a partition that respects them, ``a = 1`` (all edges within-cluster)
    and ``b = sum of (D_c / 2m)^2`` over clusters.
    """
    from omicverse.pp._champ import _modularity_coefficients

    # Two K_3 components (3 nodes each, fully connected within).
    A = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            if i != j:
                A[i, j] = 1.0
    for i in range(3, 6):
        for j in range(3, 6):
            if i != j:
                A[i, j] = 1.0
    W = sp.csr_matrix(A)
    labels = np.array([0, 0, 0, 1, 1, 1])
    a, b = _modularity_coefficients(W, labels)
    # Total weight 2m = 12 (12 directed edges, 6 undirected). Each cluster
    # contributes 6 (within edge weight). a = 12/12 = 1.0.
    assert a == pytest.approx(1.0)
    # Each cluster has total degree 6. (6/12)^2 + (6/12)^2 = 0.25 + 0.25 = 0.5.
    assert b == pytest.approx(0.5)
    # Q at γ=1 should equal a − b = 0.5 — modularity of perfect partition
    # on disconnected graph is well-known to be 0.5 for two K_3's.
    assert (a - 1.0 * b) == pytest.approx(0.5)


def test_modularity_coefficients_handle_dense_input():
    from omicverse.pp._champ import _modularity_coefficients
    W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    a, b = _modularity_coefficients(W, np.array([0, 0, 1]))
    # 2m = 6; within-cluster edge weight = 2 (only edge between 0–1);
    # a = 2/6 = 1/3.
    assert a == pytest.approx(1 / 3)


# ─────────────────────── upper hull primitive ─────────────────────────────────


def test_upper_hull_filters_dominated_points():
    from omicverse.pp._champ import _upper_hull_indices
    # Two points form a clear hull; (0.5, 0.0) is dominated.
    b = np.array([0.0, 0.5, 1.0])
    a = np.array([1.0, 0.0, 0.5])
    hull = _upper_hull_indices(b, a)
    # Point 0 (0,1) and point 2 (1,0.5) are on the upper envelope;
    # point 1 (0.5,0) is dominated. Andrew's monotone chain returns
    # them in ascending b order.
    assert hull == [0, 2]


def test_upper_hull_keeps_all_when_collinear_above():
    from omicverse.pp._champ import _upper_hull_indices
    # Three points strictly increasing in both b and a — all on hull.
    hull = _upper_hull_indices(np.array([0.0, 1.0, 2.0]),
                                np.array([0.0, 2.0, 3.0]))
    assert hull == [0, 1, 2]


# ─────────────────────────── end-to-end ───────────────────────────────────────


def test_champ_picks_a_hull_partition(adata_with_neighbors):
    """The chosen partition must be on the convex hull (never a
    dominated one)."""
    import omicverse as ov

    a = adata_with_neighbors.copy()
    _, (lo, hi), df = ov.pp.champ(
        a, n_partitions=12, gamma_min=0.1, gamma_max=2.0,
        random_state=0, verbose=False,
    )
    # Chosen γ-range was a positive interval.
    assert hi >= lo
    # The widest-range row (i.e. the chosen partition) must be on the hull.
    # The chosen partition's labels equal what's written to obs['champ'].
    chosen_n = int(a.obs["champ"].nunique())
    on_hull = df[df["on_hull"]]
    assert chosen_n in on_hull["n_clusters"].tolist()


def test_champ_writes_uns_payload(adata_with_neighbors):
    import omicverse as ov

    a = adata_with_neighbors.copy()
    ov.pp.champ(
        a, n_partitions=10, random_state=0, verbose=False,
        key_added="champ_clusters",
    )
    assert "champ_clusters" in a.obs.columns
    payload = a.uns["champ"]
    assert "Weir" in payload["method"]
    assert "chosen_n_clusters" in payload
    assert "chosen_gamma_range" in payload
    assert payload["chosen_n_clusters"] == int(a.obs["champ_clusters"].nunique())
    parts_df = pd.DataFrame(payload["partitions"])
    assert {"a", "b", "n_clusters", "on_hull",
            "gamma_lo", "gamma_hi", "gamma_range"}.issubset(parts_df.columns)


def test_champ_cleans_temp_obs_cols(adata_with_neighbors):
    import omicverse as ov

    a = adata_with_neighbors.copy()
    obs_before = set(a.obs.columns)
    ov.pp.champ(a, n_partitions=8, random_state=0, verbose=False)
    leaked = [c for c in a.obs.columns
               if c.startswith("_champ_") and c not in obs_before]
    assert leaked == []


def test_champ_records_one_provenance_entry(adata_with_neighbors):
    import omicverse as ov
    from omicverse.report._provenance import (
        clear_provenance, get_provenance,
    )

    a = adata_with_neighbors.copy()
    clear_provenance(a)
    ov.pp.champ(a, n_partitions=8, random_state=0, verbose=False)
    prov = get_provenance(a)
    names = [e["name"] for e in prov]
    assert names == ["champ"]
    e = prov[0]
    assert e["function"] == "ov.pp.champ"
    assert "CHAMP" in e["backend"]


def test_champ_requires_neighbor_graph():
    import omicverse as ov

    a = _three_blob_adata(n_per_blob=60, n_genes=20, seed=0)
    with pytest.raises(ValueError, match="connectivities"):
        ov.pp.champ(a, verbose=False)


def test_champ_widths_are_non_negative(adata_with_neighbors):
    """The clamped widths column should never report negative ranges
    even when hull partitions extend beyond gamma_max."""
    import omicverse as ov

    a = adata_with_neighbors.copy()
    _, _, df = ov.pp.champ(
        a, n_partitions=8, gamma_min=0.1, gamma_max=0.5,  # tight cap
        random_state=0, verbose=False,
    )
    hull_widths = df.loc[df["on_hull"], "gamma_range"]
    # All hull rows should report a non-negative range after clamping.
    assert (hull_widths >= 0).all()
