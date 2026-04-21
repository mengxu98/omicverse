import numpy as np
import pandas as pd
import pytest
import warnings
from anndata import AnnData

import omicverse.pl._dotplot as dotplot_mod


class _StubAnnotation:
    def __init__(self, kind, *args, **kwargs):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs


class _StubSizedHeatmap:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def add_top(self, *args, **kwargs):
        return None

    def add_left(self, *args, **kwargs):
        return None

    def add_right(self, *args, **kwargs):
        return None

    def add_dendrogram(self, *args, **kwargs):
        return None

    def add_legends(self, *args, **kwargs):
        return None

    def group_cols(self, *args, **kwargs):
        return None

    def render(self):
        return object()


@pytest.fixture
def simple_adata():
    adata = AnnData(
        np.array(
            [
                [1.0, 0.0, 2.0],
                [2.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
                [1.0, 2.0, 2.0],
            ]
        )
    )
    adata.var_names = ["g1", "g2", "g3"]
    adata.obs["cell_type"] = pd.Categorical(["A", "A", "B", "B"])
    return adata


@pytest.fixture
def stub_marsilea(monkeypatch):
    monkeypatch.setattr(dotplot_mod.ma, "SizedHeatmap", _StubSizedHeatmap)
    monkeypatch.setattr(dotplot_mod.mp, "Labels", lambda *a, **k: _StubAnnotation("Labels", *a, **k))
    monkeypatch.setattr(dotplot_mod.mp, "Colors", lambda *a, **k: _StubAnnotation("Colors", *a, **k))
    monkeypatch.setattr(dotplot_mod.mp, "Numbers", lambda *a, **k: _StubAnnotation("Numbers", *a, **k))


def test_dotplot_warns_for_broken_marsilea_056(simple_adata, stub_marsilea, monkeypatch):
    monkeypatch.setattr(dotplot_mod.ma, "__version__", "0.5.6")

    with pytest.warns(UserWarning, match="marsilea 0.5.6"):
        dotplot_mod.dotplot(
            simple_adata,
            ["g1", "g2"],
            groupby="cell_type",
            show=False,
        )


def test_dotplot_does_not_warn_for_fixed_marsilea_versions(simple_adata, stub_marsilea, monkeypatch):
    monkeypatch.setattr(dotplot_mod.ma, "__version__", "0.5.7")

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        dotplot_mod.dotplot(
            simple_adata,
            ["g1", "g2"],
            groupby="cell_type",
            show=False,
        )

    assert not [w for w in record if "marsilea 0.5.6" in str(w.message)]
