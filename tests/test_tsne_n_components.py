"""Regression test for issue #683.

Calling ``ov.pp.tsne(adata, use_rep='scaled|original|X_pca')`` blew up
with ``TypeError: tsne() got an unexpected keyword argument
'n_components'`` because the wrapper unconditionally forwarded
``n_components=2`` to ``scanpy.tl.tsne``, which does not accept it.

These tests pin three things:

1. ``ov.pp.tsne(...)`` on the default cpu (scanpy) backend no longer
   propagates ``n_components`` to ``sc.tl.tsne``.
2. Asking for ``n_components != 2`` on the cpu backend raises a clear
   ``ValueError`` (instead of the inscrutable scanpy TypeError) and
   directs the user to the torch / RAPIDS backends that actually
   support multi-dimensional t-SNE.
3. The non-scanpy code paths (``cpu-gpu-mixed``, ``gpu``) still
   receive ``n_components`` so they can produce 3-D / k-D t-SNE.

We exercise the wrapper itself (no real scanpy run) by stubbing
``sc.tl.tsne`` etc. — the bug is purely in argument forwarding.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np
import pytest
from anndata import AnnData


def _make_pca_adata(n: int = 60, n_pcs: int = 8, seed: int = 0) -> AnnData:
    """A minimal AnnData with X_pca already populated."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 12)).astype(np.float32)
    adata = AnnData(X=X)
    adata.obsm["X_pca"] = rng.standard_normal((n, n_pcs)).astype(np.float32)
    return adata


def test_cpu_tsne_does_not_forward_n_components_to_scanpy() -> None:
    """The bug: ``ov.pp.tsne`` forwarded ``n_components`` to
    ``sc.tl.tsne`` which raises TypeError. After the fix, the cpu code
    path must NOT pass ``n_components`` through."""
    import scanpy as sc
    import omicverse as ov
    from omicverse import settings

    # Capture the real scanpy signature BEFORE patching, so we can
    # cross-check that every kw we forward is a real scanpy parameter.
    real_sc_params = set(inspect.signature(sc.tl.tsne).parameters)

    adata = _make_pca_adata()
    captured: dict = {}

    def fake_sc_tsne(adata, **kw):
        captured["kw"] = kw
        return None

    prev_mode = settings.mode
    settings.mode = "cpu"
    try:
        with patch("scanpy.tl.tsne", side_effect=fake_sc_tsne):
            ov.pp.tsne(adata, use_rep="X_pca")
    finally:
        settings.mode = prev_mode

    assert "n_components" not in captured["kw"], (
        "ov.pp.tsne should not forward n_components to scanpy.tl.tsne; "
        f"actually forwarded: {sorted(captured['kw'])}"
    )
    # Every forwarded kwarg must be a real scanpy parameter.
    unknown = set(captured["kw"]) - real_sc_params
    assert not unknown, (
        f"ov.pp.tsne forwarded unknown kwargs to scanpy.tl.tsne: {unknown}"
    )


def test_cpu_tsne_with_3d_request_raises_clear_error() -> None:
    """``settings.mode='cpu'`` is scanpy-only, which is 2-D-only.
    Asking for 3-D must raise a helpful ValueError (not let scanpy
    explode with a TypeError later)."""
    import omicverse as ov
    from omicverse import settings

    adata = _make_pca_adata()

    prev_mode = settings.mode
    settings.mode = "cpu"
    try:
        with patch("scanpy.tl.tsne") as patched:
            with pytest.raises(ValueError, match=r"n_components=3"):
                ov.pp.tsne(adata, n_components=3, use_rep="X_pca")
        # The error must fire BEFORE scanpy is called.
        assert not patched.called, (
            "ov.pp.tsne should reject n_components!=2 on cpu mode "
            "before invoking scanpy"
        )
    finally:
        settings.mode = prev_mode


def test_cpu_gpu_mixed_tsne_does_forward_n_components() -> None:
    """The torch backend supports k-D t-SNE — n_components must reach
    it."""
    import omicverse as ov
    from omicverse import settings

    adata = _make_pca_adata()
    captured: dict = {}

    def fake_torch_tsne(adata, **kw):
        captured["kw"] = kw

    prev_mode = settings.mode
    settings.mode = "cpu-gpu-mixed"
    try:
        # Patch the lazily-imported helper.
        with patch("omicverse.pp._tsne.tsne", side_effect=fake_torch_tsne):
            ov.pp.tsne(adata, n_components=3, use_rep="X_pca")
    finally:
        settings.mode = prev_mode

    assert captured["kw"].get("n_components") == 3, (
        "ov.pp.tsne(cpu-gpu-mixed) must forward n_components to the "
        f"torch backend; got: {captured['kw']}"
    )
