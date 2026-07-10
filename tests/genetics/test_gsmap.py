import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse


def _run_plot_top_genes(monkeypatch, tmp_path, matrix):
    import scanpy as sc

    from omicverse.external.gsmap import diagnosis
    from omicverse.genetics._gsmap import _gsmap_runner

    adata = AnnData(
        matrix,
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1"]),
    )
    calls = []
    monkeypatch.setattr(sc.pp, "normalize_total", lambda *args, **kwargs: calls.append("normalize"))
    monkeypatch.setattr(sc.pp, "log1p", lambda *args, **kwargs: calls.append("log1p"))
    monkeypatch.setattr(
        diagnosis,
        "load_gene_diagnostic_info",
        lambda *args, **kwargs: pd.DataFrame({"Gene": ["g1"]}),
    )

    runner = _gsmap_runner(adata, workdir=str(tmp_path), sample_name="sample")
    monkeypatch.setattr(runner, "_get_latent_adata", lambda: adata)
    monkeypatch.setattr(runner, "plot_gene_gss", lambda **kwargs: kwargs["gene"])

    genes = runner.plot_top_genes("trait", top_corr_genes=1, show=False)

    return calls, genes


def test_plot_top_genes_normalizes_dense_counts(monkeypatch, tmp_path):
    calls, genes = _run_plot_top_genes(
        monkeypatch,
        tmp_path,
        np.array([[20.0], [1.0]]),
    )

    assert calls == ["normalize", "log1p"]
    assert genes == ["g1"]


def test_plot_top_genes_normalizes_sparse_counts(monkeypatch, tmp_path):
    calls, genes = _run_plot_top_genes(
        monkeypatch,
        tmp_path,
        sparse.csr_matrix([[20.0], [1.0]]),
    )

    assert calls == ["normalize", "log1p"]
    assert genes == ["g1"]
