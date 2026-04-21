"""Convex Hull of Admissible Modularity Partitions — ``ov.pp.champ``.

Implementation of Weir, Emmons, Wakefield, Hopkins & Mucha (2017),
"Post-processing partitions to identify domains of modularity
optimization" (*Algorithms* 10(3):93). The key observation is purely
geometric: for any **fixed** partition :math:`P`, Newman modularity

.. math::

   Q(\\gamma; P) =
       \\frac{1}{2m} \\sum_{ij}\\bigl[A_{ij} - \\gamma\\,
                                    \\frac{d_i d_j}{2m}\\bigr]
       \\delta(c_i, c_j)
       \\;=\\; a_P \\;-\\; \\gamma\\, b_P

is a **linear** function of the resolution parameter
:math:`\\gamma`. A family of candidate partitions therefore corresponds
to a family of lines in the :math:`(\\gamma, Q)` plane; the partition
that is modularity-optimal at *any* given :math:`\\gamma` lies on the
**upper envelope** of those lines, which is dual to the **upper convex
hull** of the points :math:`(b_P, a_P)`.

CHAMP runs leiden at many candidate :math:`\\gamma` values, computes
:math:`(a_P, b_P)` for each resulting partition, finds the upper hull,
and returns the hull partition whose **admissible γ-range** is widest
— the partition that's modularity-optimal across the broadest band of
resolutions.

The two complementary resolution-selection paths in omicverse:

- ``ov.single.auto_resolution`` — *stochastic* / *bootstrap-stability*:
  measures how reproducible each resolution's clustering is under
  data perturbation, with a null-model adjustment per Lange et al.
  2004. Picks the **most reproducible** resolution.

- ``ov.pp.champ`` — *deterministic* / *modularity-geometric*: no Monte
  Carlo, no null permutation. Picks the **most modularity-stable**
  partition by analysing the convex structure of the candidate set.

They answer different questions and can disagree; either is defensible.
"""
from __future__ import annotations

import contextlib
import io
from typing import Optional, Sequence

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._registry import register_function
from .._settings import EMOJI, add_reference
from ..report._provenance import tracked, note


# ────────────────────── modularity coefficients ───────────────────────────────


def _modularity_coefficients(W, labels) -> tuple[float, float]:
    """Return ``(a, b)`` such that Newman modularity
    :math:`Q(\\gamma; P) = a - \\gamma\\,b`.

    Works for any non-negative weighted symmetric adjacency matrix
    (sparse or dense). For undirected graphs the convention used here
    is ``2m = W.sum()`` (each edge counted twice in the symmetric
    matrix).

    Linear in ``nnz(W)``: iterates the COO entries once for ``a`` and
    does a single scatter-add over labels for ``b``.
    """
    if not sp.issparse(W):
        W = sp.csr_matrix(W)
    coo = W.tocoo()
    total = float(W.sum())  # 2m
    if total <= 0:
        return 0.0, 0.0
    d = np.asarray(W.sum(axis=1)).ravel()

    # a = (1/2m) * within-cluster edge weight
    same = labels[coo.row] == labels[coo.col]
    a = float(coo.data[same].sum()) / total

    # b = (1/(2m)^2) * Σ_c (Σ_{i in c} d_i)^2
    unique_labels, inv = np.unique(labels, return_inverse=True)
    D = np.zeros(len(unique_labels), dtype=np.float64)
    np.add.at(D, inv, d)
    b = float(np.sum(D * D)) / (total * total)
    return a, b


# ────────────────────── upper convex hull ──────────────────────────────────────


def _upper_hull_indices(b: np.ndarray, a: np.ndarray) -> list[int]:
    """Andrew's monotone-chain upper-hull pass.

    Returns the indices of points lying on the **upper** convex hull of
    the 2-D points ``(b_i, a_i)``, in order of ascending ``b``.
    Geometrically dual to the upper envelope of the lines
    :math:`y = a_i - b_i x`.
    """
    n = len(b)
    if n <= 2:
        return list(range(n))
    # Sort by b ascending; ties broken by a descending so the very
    # first point at each b-coordinate is the highest.
    order = sorted(range(n), key=lambda i: (b[i], -a[i]))
    hull: list[int] = []
    for i in order:
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            # Cross product of (p2 - p1) × (p3 - p1).
            cross = ((b[i2] - b[i1]) * (a[i] - a[i1]) -
                      (a[i2] - a[i1]) * (b[i] - b[i1]))
            # Upper hull keeps RIGHT turns (cross < 0); pop otherwise.
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


# ───────────────────────────── public entry ───────────────────────────────────


@register_function(
    aliases=['CHAMP', 'champ', 'modularity convex hull',
             'admissible partitions'],
    category="preprocessing",
    description=(
        "Convex Hull of Admissible Modularity Partitions (Weir et al. "
        "2017). Generates candidate Leiden partitions across a γ grid, "
        "computes the (a, b) coefficients of Q(γ) = a − γ·b for each, "
        "finds their upper convex hull (the partitions modularity-"
        "optimal at *some* γ), and returns the hull partition with the "
        "widest admissible γ-range — the deterministic modularity-"
        "geometric counterpart to ov.single.auto_resolution's "
        "stochastic null-adjusted bootstrap stability."
    ),
    prerequisites={'functions': ['pp.neighbors']},
    requires={'obsp': ['connectivities']},
    produces={'obs': ['champ'], 'uns': ['champ']},
    auto_fix='auto',
    examples=[
        'partition, gamma_range, df = ov.pp.champ(adata)',
        'ov.pp.champ(adata, resolutions=np.linspace(0.05, 3.0, 30))',
    ],
    related=['pp.leiden', 'single.auto_resolution'],
)
@tracked('champ', 'ov.pp.champ')
def champ(
    adata: anndata.AnnData,
    resolutions: Optional[Sequence[float]] = None,
    *,
    n_partitions: int = 30,
    gamma_min: float = 0.05,
    gamma_max: float = 3.0,
    key_added: str = 'champ',
    random_state: int = 0,
    verbose: bool = True,
):
    r"""Pick the modularity-stablest Leiden partition via CHAMP
    (Weir et al. 2017).

    Algorithm
    ---------
    1. Run Leiden at :paramref:`n_partitions` candidate γ values
       evenly spaced over :math:`[\gamma_\min,\,\gamma_\max]` (or use
       the explicit grid passed in :paramref:`resolutions`). Deduplicate
       partitions that produce identical labels.
    2. For each unique partition :math:`P`, compute
       :math:`(a_P, b_P)` such that

       .. math::

           Q(\gamma; P) = a_P - \gamma\, b_P.

    3. Compute the **upper convex hull** of the points
       :math:`\{(b_P, a_P)\}`. The hull's vertices are the partitions
       that are modularity-optimal for *some* γ; everything else is
       dominated.
    4. For each consecutive pair of hull vertices, compute the γ at
       which their lines intersect:

       .. math::

           \gamma_{i,i+1} = \frac{a_{i+1} - a_i}{b_{i+1} - b_i}.

    5. Each hull partition's **admissible γ-range** is the interval
       between its two adjacent crossover γ's (capped at
       :math:`[0,\,\gamma_\max]`).
    6. Return the hull partition with the **widest admissible range**.

    No Monte Carlo, no null model — purely a geometric statement about
    the modularity landscape's convex structure.

    Parameters
    ----------
    adata
        AnnData with a precomputed neighbor graph
        (``adata.obsp['connectivities']``).
    resolutions
        Explicit γ grid; overrides :paramref:`n_partitions` /
        :paramref:`gamma_min` / :paramref:`gamma_max`.
    n_partitions
        Number of γ values to scan when ``resolutions`` is ``None``.
    gamma_min, gamma_max
        Endpoints of the γ scan; ``gamma_max`` also caps the leftmost
        hull partition's "open" admissible range so widths are finite.
    key_added
        ``adata.obs`` column to write the chosen partition's labels to.
    random_state
        Seed for Leiden RNG.
    verbose
        Stream per-step progress.

    Returns
    -------
    Tuple[anndata.AnnData, Tuple[float, float], pandas.DataFrame]
        ``(adata, (gamma_lo, gamma_hi), partitions_df)`` where
        ``partitions_df`` is one row per unique candidate partition with
        columns ``a``, ``b``, ``n_clusters``, ``origin_resolution``,
        ``on_hull`` (bool), ``gamma_lo``, ``gamma_hi``,
        ``gamma_range``. Also writes ``adata.obs[key_added]`` and
        ``adata.uns['champ']``.

    References
    ----------
    Weir, Emmons, Wakefield, Hopkins, Mucha. "Post-processing partitions
    to identify domains of modularity optimization."
    *Algorithms* 10(3):93, 2017. https://doi.org/10.3390/a10030093
    """
    if 'connectivities' not in adata.obsp:
        raise ValueError(
            "champ requires a precomputed neighbor graph "
            "(adata.obsp['connectivities']); run ov.pp.neighbors first."
        )
    if not (0 <= gamma_min < gamma_max):
        raise ValueError(
            f"gamma_min/gamma_max must satisfy 0 <= min < max; got "
            f"({gamma_min}, {gamma_max})."
        )

    if resolutions is None:
        resolutions = list(np.linspace(gamma_min, gamma_max, n_partitions))
    resolutions = sorted(float(np.round(r, 4)) for r in resolutions)

    if verbose:
        print(f"{EMOJI['start']} CHAMP: scanning {len(resolutions)} "
               f"resolutions ∈ [{gamma_min:.2f}, {gamma_max:.2f}]")

    # 1. Generate candidate partitions, deduplicating identical ones.
    from ..pp import leiden as _leiden  # tracked; nesting guard silences

    TMP_KEY = '_champ_tmp'
    partitions: list[dict] = []
    seen_signatures: dict[tuple, int] = {}
    for r in resolutions:
        with contextlib.redirect_stdout(io.StringIO()):
            _leiden(adata, resolution=r, key_added=TMP_KEY,
                    random_state=random_state)
        labels = adata.obs[TMP_KEY].astype(int).values.copy()
        sig = tuple(labels.tolist())
        if sig in seen_signatures:
            continue
        seen_signatures[sig] = len(partitions)
        partitions.append({
            'origin_resolution': r,
            'labels': labels,
            'n_clusters': int(np.unique(labels).size),
        })
    if TMP_KEY in adata.obs.columns:
        del adata.obs[TMP_KEY]

    if verbose:
        print(f"  {len(partitions)} unique partitions "
               f"(from {len(resolutions)} resolutions)")

    # 2. Compute (a, b) for each unique partition.
    W = adata.obsp['connectivities']
    bs = np.empty(len(partitions))
    as_ = np.empty(len(partitions))
    for i, p in enumerate(partitions):
        a, b = _modularity_coefficients(W, p['labels'])
        as_[i] = a
        bs[i] = b
        p['a'] = a
        p['b'] = b

    # 3. Upper convex hull in (b, a) plane.
    hull_idx = _upper_hull_indices(bs, as_)
    H = len(hull_idx)
    if verbose:
        print(f"  {H} partitions on the upper convex hull (admissible)")

    # 4. Crossover γ between consecutive hull partitions.
    crossovers = []
    for k in range(H - 1):
        i, j = hull_idx[k], hull_idx[k + 1]
        denom = bs[j] - bs[i]
        crossovers.append(float((as_[j] - as_[i]) / denom)
                            if denom != 0 else float('inf'))

    # 5. Per-hull-partition admissible γ-range.
    on_hull = np.zeros(len(partitions), dtype=bool)
    on_hull[hull_idx] = True
    gamma_lo = np.full(len(partitions), np.nan)
    gamma_hi = np.full(len(partitions), np.nan)
    for k, hi_pos in enumerate(hull_idx):
        if H == 1:
            lo, hi = 0.0, gamma_max
        elif k == 0:
            # Leftmost (smallest b): admissible at γ > crossovers[0]
            lo = crossovers[0]
            hi = gamma_max
        elif k == H - 1:
            # Rightmost (largest b): admissible at γ < crossovers[-1]
            lo = 0.0
            hi = crossovers[-1]
        else:
            # Middle: bracketed by two crossovers
            lo = crossovers[k]
            hi = crossovers[k - 1]
        gamma_lo[hi_pos] = max(0.0, lo)
        gamma_hi[hi_pos] = min(gamma_max, hi)

    # 6. Pick the widest range. Clamp negative widths (hull partitions
    # whose admissible region extends beyond gamma_max get 0, not a
    # confusing negative number) and exclude non-hull rows from the
    # argmax via -inf.
    raw_widths = gamma_hi - gamma_lo
    widths = np.where(on_hull,
                       np.maximum(raw_widths, 0.0),
                       np.full(len(partitions), -np.inf))
    best_idx = int(np.argmax(widths))
    best_partition = partitions[best_idx]
    best_lo = float(gamma_lo[best_idx])
    best_hi = float(gamma_hi[best_idx])

    # ── Write back ──────────────────────────────────────────────────
    adata.obs[key_added] = pd.Categorical(
        best_partition['labels'].astype(str)
    )
    df = pd.DataFrame({
        'origin_resolution': [p['origin_resolution'] for p in partitions],
        'a':                  as_,
        'b':                  bs,
        'n_clusters':         [p['n_clusters'] for p in partitions],
        'on_hull':            on_hull,
        'gamma_lo':           gamma_lo,
        'gamma_hi':           gamma_hi,
        'gamma_range':        widths,
    }).sort_values('b').reset_index(drop=True)

    adata.uns['champ'] = {
        'method': 'CHAMP (Weir et al. 2017)',
        'partitions':            df.to_dict('list'),
        'chosen_origin_resolution': float(best_partition['origin_resolution']),
        'chosen_n_clusters':     int(best_partition['n_clusters']),
        'chosen_gamma_range':    [best_lo, best_hi],
        'chosen_gamma_width':    best_hi - best_lo,
        'gamma_min':             float(gamma_min),
        'gamma_max':             float(gamma_max),
    }
    add_reference(
        adata, 'champ',
        f'CHAMP partition (γ-range [{best_lo:.3f}, {best_hi:.3f}], '
        f'{best_partition["n_clusters"]} clusters)',
    )

    if verbose:
        print(f"{EMOJI['done']} chosen partition: "
               f"{best_partition['n_clusters']} clusters; admissible "
               f"γ ∈ [{best_lo:.3f}, {best_hi:.3f}] "
               f"(width {best_hi - best_lo:.3f})")

    note(
        backend=f'CHAMP · {len(partitions)} partitions · '
                 f'{H} on hull',
        viz=[
            {'function': 'ov.pl.cluster_sizes_bar',
              'kwargs': {'groupby': key_added}},
            *([{'function': 'ov.pl.embedding',
                 'kwargs': {'basis': 'X_umap', 'color': key_added,
                            'frameon': 'small'}}]
               if 'X_umap' in adata.obsm else []),
        ],
    )

    return adata, (best_lo, best_hi), df
