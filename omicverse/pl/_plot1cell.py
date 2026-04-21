"""Python port of plot1cell::plot_circlize (HaojiaWu/plot1cell).

The original R function arranges a UMAP/t-SNE inside a circular
"circlize" layout: the circumference is split into cluster sectors
(arc length ~ log10(n_cells)), and any number of concentric outer
rings show per-cell metadata grouped by cluster. The scatter itself
is drawn in Cartesian space inside the unit circle, so the UMAP
stays readable — only the *annotation* part becomes circular.

This port uses only matplotlib + scipy.stats.gaussian_kde; no R
dependency, no circlize.
"""
from __future__ import annotations

import warnings
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Wedge
from scipy.stats import gaussian_kde

from .._registry import register_function
from ._palette import sc_color, pastel_palette, vibrant_palette


# ------------------------------------------------------------------ #
# Small helpers that mirror the R names so the mapping stays obvious  #
# ------------------------------------------------------------------ #
def _transform_coordinates(v: np.ndarray, zoom: float) -> np.ndarray:
    """R: ``transform_coordinates``. Centre ``v`` at the midpoint of
    its extent and rescale so the absolute max equals ``zoom``. The
    two axes are transformed **independently** (same as the R code),
    which deliberately fills the circle instead of preserving the
    original UMAP aspect ratio."""
    lo, hi = float(np.min(v)), float(np.max(v))
    centred = v - 0.5 * (lo + hi)
    m = float(np.max(np.abs(centred))) or 1.0
    return centred * zoom / m


def _cluster_order(clusters: Sequence[str]) -> List[str]:
    """Preserve pandas-categorical order when given, else first-seen."""
    if isinstance(clusters, pd.Categorical):
        return list(clusters.categories)
    cats = pd.Series(list(clusters)).astype("category")
    return list(cats.cat.categories)


def _resolve_palette(n: int, palette) -> List[str]:
    if palette is None:
        if n <= len(sc_color):
            return list(sc_color[:n])
        cmap = plt.get_cmap("tab20", n)
        return [cmap(i) for i in range(n)]
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette, n)
        return [cmap(i) for i in range(n)]
    palette = list(palette)
    if len(palette) < n:
        # Recycle
        palette = [palette[i % len(palette)] for i in range(n)]
    return palette[:n]


def _run_length(vals: np.ndarray):
    """Return ``(starts, lengths, values)`` of consecutive runs."""
    if len(vals) == 0:
        return np.array([]), np.array([]), np.array([])
    change = np.concatenate([[True], vals[1:] != vals[:-1]])
    starts = np.flatnonzero(change)
    lengths = np.diff(np.concatenate([starts, [len(vals)]]))
    values = vals[starts]
    return starts, lengths, values


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #
@register_function(
    aliases=[
        "plot1cell",
        "circlize_umap",
        "circular_umap",
        "环形UMAP",
        "圆形UMAP",
    ],
    category="pl",
    description=(
        "Circular UMAP/t-SNE with concentric per-cell metadata tracks "
        "(Python port of R plot1cell::plot_circlize)."
    ),
    examples=[
        "# basic: clusters only",
        "ov.pl.plot1cell(adata, clusters='leiden', basis='X_umap')",
        "# with extra metadata tracks",
        "ov.pl.plot1cell(adata, clusters='cell_type',",
        "                tracks=['sample', 'phase'])",
    ],
    related=["pl.embedding", "pl.umap", "pl.embedding_atlas"],
)
def plot1cell(
    adata,
    clusters: str,
    basis: str = "X_umap",
    tracks: Optional[Sequence[str]] = None,
    *,
    coord_scale: float = 0.8,
    point_size: float = 3.0,
    point_alpha: float = 0.3,
    contour_levels: Optional[Sequence[float]] = (0.2, 0.3),
    contour_color: str = "#ae9c76",
    kde_n: int = 200,
    do_label: bool = True,
    label_fontsize: float = 9.0,
    label_orient: str = "auto",
    cluster_palette=None,
    track_palette=None,
    bg_color: str = "#F9F2E4",
    gap_between_deg: float = 2.0,
    gap_start_deg: float = 12.0,
    cluster_track_width: float = 0.035,
    track_width: float = 0.025,
    track_gap: float = 0.004,
    cluster_label_pad: float = 0.08,
    figsize=(7, 7),
    ax: Optional[Axes] = None,
    show: bool = True,
    return_data: bool = False,
):
    """Circular UMAP with metadata tracks.

    Mirrors plot1cell::plot_circlize (Wu 2021). Clusters are laid out
    as arc sectors on the circumference, sector length proportional
    to ``log10(n_cells)``. The scatter is drawn in Cartesian
    coordinates **inside** the unit circle, with a Gaussian-KDE
    contour overlay. Each entry in ``tracks`` becomes one extra
    concentric ring coloured by the run-length segments of that
    metadata column **within each cluster sector**.

    Parameters
    ----------
    adata : AnnData
    clusters : str
        Key in ``adata.obs`` giving the cluster label per cell.
    basis : str
        Key in ``adata.obsm``; first two columns are used.
    tracks : list[str] | None
        Additional ``adata.obs`` columns to show as outer rings.
    coord_scale : float
        ``zoom`` parameter from the R version; the scatter fills
        ``[-coord_scale, coord_scale]`` on each axis.
    contour_levels : tuple[float, ...] | None
        Level set(s) of the 2-D KDE to overlay. ``None`` disables
        contours.
    gap_between_deg, gap_start_deg : float
        Angular gaps (degrees) between adjacent cluster sectors and
        at the circle's starting point.
    cluster_track_width, track_width : float
        Radial thickness of the cluster ring and each metadata ring,
        expressed as a fraction of the unit radius.
    ax : matplotlib Axes | None
        If given, draw into it (must be ``aspect='equal'``). A new
        figure is created otherwise.
    return_data : bool
        If True, also return the per-cell dataframe used for
        plotting (useful for reproducing or extending the figure).

    Returns
    -------
    ax : matplotlib.axes.Axes
    (ax, df) if ``return_data=True``.
    """
    # --- 1. Fetch + validate inputs -------------------------------
    if basis not in adata.obsm:
        raise KeyError(f"basis {basis!r} not in adata.obsm")
    if clusters not in adata.obs:
        raise KeyError(f"clusters column {clusters!r} not in adata.obs")
    tracks = list(tracks) if tracks else []
    for t in tracks:
        if t not in adata.obs:
            raise KeyError(f"track column {t!r} not in adata.obs")

    xy = np.asarray(adata.obsm[basis])[:, :2]
    cl_vals = adata.obs[clusters].astype(str).to_numpy()
    cl_order = _cluster_order(adata.obs[clusters])

    # Transform to the plotting square
    x = _transform_coordinates(xy[:, 0], coord_scale)
    y = _transform_coordinates(xy[:, 1], coord_scale)

    df = pd.DataFrame({
        "cell": np.asarray(adata.obs_names),
        "cluster": cl_vals,
        "x": x,
        "y": y,
    })
    for t in tracks:
        df[t] = adata.obs[t].astype(str).to_numpy()

    # Within each cluster, rank 1..N. x_polar2 = log10(rank) is the
    # *width* assigned to each cell along its sector (same as R).
    df["x_polar"] = df.groupby("cluster").cumcount() + 1

    # --- 2. Sector geometry ---------------------------------------
    # angle span of each cluster is proportional to log10(n_cluster),
    # matching circlize's x-axis = x_polar2.
    counts = df["cluster"].value_counts().reindex(cl_order).fillna(0).astype(int)
    log_counts = np.log10(counts.clip(lower=1).to_numpy())
    n_clusters = len(cl_order)
    total_gap = (n_clusters - 1) * gap_between_deg + gap_start_deg
    usable = 360.0 - total_gap
    if usable <= 0:
        raise ValueError(
            f"gaps ({total_gap}°) exceed 360°; reduce gap_between_deg / "
            f"gap_start_deg or collapse clusters."
        )
    cluster_deg = usable * log_counts / log_counts.sum()

    # Start just past 3-o'clock going counter-clockwise: matches the
    # circlize default where the start-gap sits at angle 0 (east).
    sector_bounds: dict[str, tuple[float, float]] = {}
    theta = 0.5 * gap_start_deg
    for cl, deg in zip(cl_order, cluster_deg):
        a0 = theta
        a1 = theta + deg
        sector_bounds[cl] = (a0, a1)
        theta = a1 + gap_between_deg

    # --- 3. Figure setup ------------------------------------------
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure
    ax.set_aspect("equal")
    outer = 1.0 + cluster_track_width + len(tracks) * (track_width + track_gap) + 0.02
    ax.set_xlim(-outer, outer)
    ax.set_ylim(-outer, outer)
    ax.set_facecolor(bg_color)
    if created_fig:
        fig.patch.set_facecolor(bg_color)
    ax.set_axis_off()

    # --- 4. Scatter + KDE -----------------------------------------
    cluster_colors = _resolve_palette(n_clusters, cluster_palette)
    cl_to_color = dict(zip(cl_order, cluster_colors))
    pt_colors = [cl_to_color[c] for c in df["cluster"]]
    ax.scatter(
        df["x"].to_numpy(), df["y"].to_numpy(),
        s=point_size, c=pt_colors, alpha=point_alpha,
        linewidths=0, zorder=2,
    )

    if contour_levels is not None and len(df) >= 10:
        try:
            kde = gaussian_kde(np.vstack([df["x"].to_numpy(), df["y"].to_numpy()]))
            lim = coord_scale * 1.05
            gx, gy = np.mgrid[-lim:lim:complex(kde_n), -lim:lim:complex(kde_n)]
            zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            # Normalise to match the R ``levels`` interpretation (values
            # are roughly the density relative to its max over the grid).
            zz = zz / (zz.max() + 1e-12)
            ax.contour(gx, gy, zz, levels=sorted(contour_levels),
                       colors=contour_color, linewidths=0.8, zorder=3)
        except Exception as exc:  # pragma: no cover — KDE can fail on tiny data
            warnings.warn(f"KDE contour skipped: {exc}")

    # --- 5. Cluster ring ------------------------------------------
    r_in = 1.0
    r_out = 1.0 + cluster_track_width
    for cl, (a0, a1) in sector_bounds.items():
        w = Wedge(
            center=(0, 0), r=r_out, theta1=a0, theta2=a1,
            width=r_out - r_in, facecolor=cl_to_color[cl],
            edgecolor="none", linewidth=0, zorder=4,
        )
        ax.add_patch(w)

        if do_label:
            mid = 0.5 * (a0 + a1) % 360.0
            rad = np.deg2rad(mid)
            # 'auto': tangent labels for few clusters (they fit the
            # sector), radial for many (tangent text would collide).
            orient = label_orient
            if orient == "auto":
                orient = "tangent" if n_clusters <= 10 else "radial"
            if orient == "tangent":
                lx = (r_out + cluster_label_pad) * np.cos(rad)
                ly = (r_out + cluster_label_pad) * np.sin(rad)
                rot = mid - 90.0
                if mid > 180.0:
                    rot += 180.0
                ax.text(lx, ly, str(cl), ha="center", va="center",
                        rotation=rot, rotation_mode="anchor",
                        fontsize=label_fontsize, zorder=5)
            else:
                # Radial: each label is a spoke — angular footprint is
                # just one line-height, so they never collide even with
                # 30+ clusters. On the left half we flip by 180° so the
                # text still reads left-to-right when viewed upright.
                flip = 90.0 < mid < 270.0
                rot = (mid - 180.0) if flip else mid
                ha = "right" if flip else "left"
                lx = (r_out + cluster_label_pad * 0.4) * np.cos(rad)
                ly = (r_out + cluster_label_pad * 0.4) * np.sin(rad)
                ax.text(lx, ly, str(cl), ha=ha, va="center",
                        rotation=rot, rotation_mode="anchor",
                        fontsize=label_fontsize, zorder=5)

    # --- 6. In-UMAP cluster labels --------------------------------
    if do_label:
        import matplotlib.patheffects as pe
        texts = []
        for cl in cl_order:
            sub = df[df["cluster"] == cl]
            texts.append(ax.text(
                sub["x"].median(), sub["y"].median(), str(cl),
                ha="center", va="center", fontsize=label_fontsize,
                zorder=6,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            ))
        try:
            from adjustText import adjust_text
            adjust_text(
                texts, ax=ax,
                expand=(1.1, 1.2),
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.5),
                only_move={"text": "xy"},
            )
        except ImportError:
            # adjustText is optional — silently skip repel if unavailable.
            pass

    # --- 7. Metadata tracks ---------------------------------------
    for t_idx, t in enumerate(tracks):
        r0 = r_out + track_gap + t_idx * (track_width + track_gap)
        r1 = r0 + track_width
        levels = df[t].unique().tolist()
        # Stable ordering (categorical if possible, else sorted)
        try:
            levels = list(adata.obs[t].astype("category").cat.categories)
        except Exception:
            levels = sorted(levels)
        colors = _resolve_palette(len(levels), track_palette)
        lvl_to_color = dict(zip(levels, colors))

        for cl in cl_order:
            a0, a1 = sector_bounds[cl]
            sub = df[df["cluster"] == cl].sort_values(t)
            vals = sub[t].to_numpy()
            _, lengths, rleg_vals = _run_length(vals)
            total = len(sub)
            if total == 0:
                continue
            cum = 0
            for length, val in zip(lengths, rleg_vals):
                ang0 = a0 + (cum / total) * (a1 - a0)
                ang1 = a0 + ((cum + length) / total) * (a1 - a0)
                ax.add_patch(Wedge(
                    center=(0, 0), r=r1, theta1=ang0, theta2=ang1,
                    width=r1 - r0, facecolor=lvl_to_color[val],
                    edgecolor="none", linewidth=0, zorder=4,
                ))
                cum += length

        # Track label inside the start-gap, stacked radially so
        # multiple tracks don't collide. (The start gap is centred on
        # angle 0 — the right side of the circle.)
        gap_mid_rad = np.deg2rad(0.0)
        mid_r = 0.5 * (r0 + r1)
        ax.text(
            np.cos(gap_mid_rad) * (mid_r + 0.005),
            np.sin(gap_mid_rad) * (mid_r + 0.005),
            t, ha="center", va="center",
            fontsize=label_fontsize * 0.75,
            rotation=-90, rotation_mode="anchor",
            zorder=5,
        )

    # --- 8. Track legends -----------------------------------------
    if tracks:
        legend_handles = []
        for t_idx, t in enumerate(tracks):
            try:
                levels = list(adata.obs[t].astype("category").cat.categories)
            except Exception:
                levels = sorted(df[t].unique().tolist())
            colors = _resolve_palette(len(levels), track_palette)
            for lvl, col in zip(levels, colors):
                legend_handles.append(
                    plt.Line2D([], [], marker="s", linestyle="",
                               markerfacecolor=col, markeredgecolor=col,
                               markersize=8, label=f"{t}={lvl}")
                )
        ax.legend(
            handles=legend_handles, loc="center left",
            bbox_to_anchor=(1.02, 0.5), frameon=False,
            fontsize=label_fontsize * 0.8,
        )

    if show and created_fig:
        plt.show()

    if return_data:
        return ax, df
    return ax
