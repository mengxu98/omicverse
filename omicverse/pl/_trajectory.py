"""Trajectory visualizations shared by trajectory inference backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from scipy import sparse

from .._registry import register_function
from ._single import embedding


@dataclass
class _GraphData:
    node_coords: np.ndarray
    edges: list[tuple[int, int]]
    branch_indices: list[int]
    branch_labels: list[str]
    cell_coords: Optional[np.ndarray] = None
    edge_weights: Optional[list[float]] = None
    directed_edges: Optional[list[tuple[int, int, float]]] = None


def _basis_key(adata, basis):
    if basis in adata.obsm:
        return basis
    if isinstance(basis, str) and not basis.startswith("X_"):
        candidate = f"X_{basis}"
        if candidate in adata.obsm:
            return candidate
    raise KeyError(f"`{basis}` was not found in `adata.obsm`.")


def _normalize_method(adata, method="auto", model=None):
    if method is None or method == "auto":
        if "monocle" in adata.uns:
            return "monocle"
        if "paga" in adata.uns:
            return "paga"
        if model is not None and hasattr(model, "curves"):
            return "slingshot"
        raise ValueError(
            "Could not infer trajectory method. Pass `method='monocle'`, "
            "`method='paga'`, or provide a supported model."
        )
    return str(method).lower()


def _default_basis(method):
    if method == "monocle":
        return "X_DDRTree"
    return "X_umap"


def _is_categorical(values):
    values = pd.Series(values)
    return (
        isinstance(values.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(values)
        or pd.api.types.is_string_dtype(values)
        or pd.api.types.is_bool_dtype(values)
    )


def _get_obs_color_map(adata, color_key, values):
    series = pd.Series(values)
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = pd.Index(series.dtype.categories)
    else:
        categories = pd.Index(pd.unique(series))

    uns_key = f"{color_key}_colors"
    uns_colors = adata.uns.get(uns_key)
    if uns_colors is not None and len(uns_colors) >= len(categories):
        return {cat: uns_colors[idx] for idx, cat in enumerate(categories)}

    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(cycle_colors) >= len(categories):
        return {
            cat: mcolors.to_hex(cycle_colors[idx], keep_alpha=True)
            for idx, cat in enumerate(categories)
        }

    from ._palette import palette_112, palette_56, sc_color

    if len(categories) <= len(sc_color):
        palette = sc_color
    elif len(categories) <= len(palette_56):
        palette = palette_56
    else:
        palette = palette_112
    return {
        cat: mcolors.to_hex(palette[idx % len(palette)], keep_alpha=True)
        for idx, cat in enumerate(categories)
    }


def _build_categorical_legend(
    ax,
    color_map,
    *,
    title=None,
    max_cols=6,
    anchor="right margin",
    loc=None,
    fontsize=None,
    markersize=6,
):
    ncols = min(max_cols, max(1, int(np.ceil(np.sqrt(len(color_map))))))
    if anchor in {"right margin", "right"}:
        loc = "center left" if loc is None else loc
        bbox_to_anchor = (1.02, 0.5)
        ncols = 1
    else:
        loc = "lower center" if loc is None else loc
        bbox_to_anchor = anchor
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_map[state],
            markeredgecolor="none",
            markersize=markersize,
            label=str(state),
        )
        for state in color_map.keys()
    ]
    legend = ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncols,
        frameon=False,
        title=title,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.2,
    )
    if fontsize is not None:
        if legend is not None and legend.get_title() is not None:
            title_fontsize = (
                fontsize + 0.5 if isinstance(fontsize, (int, float)) else fontsize
            )
            legend.get_title().set_fontsize(title_fontsize)
        for text in legend.get_texts():
            text.set_fontsize(fontsize)
    return legend


def _add_axis_padding(ax, coords_x, coords_y, frac=0.04):
    if len(coords_x) == 0 or len(coords_y) == 0:
        return
    dx = np.nanmax(coords_x) - np.nanmin(coords_x)
    dy = np.nanmax(coords_y) - np.nanmin(coords_y)
    dx = dx if dx > 0 else 1.0
    dy = dy if dy > 0 else 1.0
    ax.set_xlim(np.nanmin(coords_x) - dx * frac, np.nanmax(coords_x) + dx * frac)
    ax.set_ylim(np.nanmin(coords_y) - dy * frac, np.nanmax(coords_y) + dy * frac)


def _add_branch_point(
    ax,
    x,
    y,
    label,
    *,
    size=150,
    facecolor="#6F6F6F",
    edgecolor="white",
    text_color="white",
    fontsize=10,
    alpha=0.95,
    zorder=6,
    clip_on=False,
):
    ax.scatter(
        x,
        y,
        s=size,
        c=facecolor,
        zorder=zorder,
        edgecolors=edgecolor,
        linewidths=1.2,
        alpha=alpha,
        clip_on=clip_on,
    )
    ax.text(
        x,
        y,
        str(label),
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight="bold",
        zorder=zorder + 1,
        path_effects=[pe.withStroke(linewidth=1.5, foreground=facecolor)],
        clip_on=clip_on,
    )


def _branch_label_radius_px(*, size, fontsize, dpi):
    marker_radius = np.sqrt(max(float(size), 1.0) / np.pi) * dpi / 72.0
    text_radius = float(fontsize) * dpi / 72.0 * 0.65
    return max(marker_radius, text_radius) + 3.0


def _resolve_branch_label_positions(
    ax,
    anchors_data,
    *,
    size,
    fontsize,
    adjust=True,
    avoid_data=None,
):
    anchors_data = np.asarray(anchors_data, dtype=float)
    if len(anchors_data) == 0:
        return anchors_data, np.zeros(0, dtype=bool)
    if not adjust or len(anchors_data) == 1:
        return anchors_data, np.zeros(len(anchors_data), dtype=bool)

    anchors_px = ax.transData.transform(anchors_data)
    radius = _branch_label_radius_px(
        size=size,
        fontsize=fontsize,
        dpi=ax.figure.dpi,
    )
    min_gap = radius * 2.0 + 4.0
    move_mask = np.zeros(len(anchors_px), dtype=bool)
    for i in range(len(anchors_px)):
        for j in range(i + 1, len(anchors_px)):
            if np.linalg.norm(anchors_px[i] - anchors_px[j]) < min_gap:
                move_mask[[i, j]] = True
    if not np.any(move_mask):
        return anchors_data, move_mask

    avoid_px = (
        ax.transData.transform(np.asarray(avoid_data, dtype=float))
        if avoid_data is not None and len(avoid_data) > 0
        else anchors_px
    )
    plot_center = np.nanmean(avoid_px, axis=0)
    directions = np.array(
        [
            [-1.00, 0.00],
            [-0.70, 0.70],
            [0.00, 1.00],
            [0.70, 0.70],
            [1.00, 0.00],
            [0.70, -0.70],
            [0.00, -1.00],
            [-0.70, -0.70],
        ],
        dtype=float,
    )
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    offset = max(radius * 4.0, 60.0)

    label_positions = anchors_px.copy()
    placed_px = [anchors_px[i] for i in range(len(anchors_px)) if not move_mask[i]]
    for idx, anchor in enumerate(anchors_px):
        if not move_mask[idx]:
            continue
        outward = anchor - plot_center
        norm = np.linalg.norm(outward)
        outward = outward / norm if norm > 1e-6 else np.array([0.0, 1.0])
        best_score = np.inf
        best_center = anchor
        for direction in directions:
            center = anchor + direction * offset
            label_penalty = 0.0
            for placed in placed_px:
                distance = np.linalg.norm(center - placed)
                label_penalty += max(0.0, 56.0 - distance) ** 2
            anchor_dist = np.linalg.norm(center - anchors_px, axis=1)
            anchor_penalty = np.sum(np.maximum(0.0, 36.0 - anchor_dist) ** 2)
            avoid_dist = np.linalg.norm(center - avoid_px, axis=1)
            avoid_penalty = max(0.0, 24.0 - np.nanmin(avoid_dist)) ** 2
            outward_penalty = -18.0 * float(np.dot(direction, outward))
            score = (
                label_penalty * 4.0
                + anchor_penalty * 2.0
                + avoid_penalty
                + outward_penalty
            )
            if score < best_score:
                best_score = score
                best_center = center
        label_positions[idx] = best_center
        placed_px.append(best_center)

    placed_data = ax.transData.inverted().transform(label_positions)
    moved = move_mask & (np.linalg.norm(label_positions - anchors_px, axis=1) > 1.0)
    return placed_data, moved


def _draw_branch_points(
    ax,
    anchors,
    labels,
    *,
    size=150,
    facecolor="#6F6F6F",
    text_color="white",
    fontsize=10,
    alpha=0.95,
    zorder=6,
    adjust_labels=True,
    avoid_coords=None,
):
    anchors = np.asarray(anchors, dtype=float)
    labels = [str(label) for label in labels]
    if len(anchors) == 0:
        return []

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    label_positions, moved = _resolve_branch_label_positions(
        ax,
        anchors,
        size=size,
        fontsize=fontsize,
        adjust=adjust_labels,
        avoid_data=avoid_coords,
    )
    for anchor, position, label, is_moved in zip(
        anchors, label_positions, labels, moved
    ):
        if is_moved:
            ax.plot(
                [anchor[0], position[0]],
                [anchor[1], position[1]],
                color="#C8C8C8",
                linewidth=0.7,
                alpha=0.85,
                linestyle=(0, (2, 2)),
                zorder=zorder - 1,
                solid_capstyle="round",
                clip_on=False,
            )
        _add_branch_point(
            ax,
            position[0],
            position[1],
            label,
            size=size,
            facecolor=facecolor,
            text_color=text_color,
            fontsize=fontsize,
            alpha=alpha,
            zorder=zorder,
            clip_on=False,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return label_positions


def _monocle_graph_data(adata, *, x=0, y=1, theta=0.0):
    if "monocle" not in adata.uns:
        raise KeyError("adata.uns['monocle'] is missing.")
    monocle = adata.uns["monocle"]
    if monocle.get("dim_reduce_type", "DDRTree") != "DDRTree":
        raise ValueError(
            "Monocle trajectory plotting expects a DDRTree reduction, got "
            f"{monocle.get('dim_reduce_type')!r}."
        )

    node_coords = np.asarray(monocle["reducedDimK"]).T
    cell_coords = np.asarray(monocle.get("reducedDimS", np.empty((0, 0)))).T
    if theta != 0:
        rot = _rotation_matrix(theta)
        node_coords = node_coords[:, [x, y]] @ rot.T
        cell_coords = cell_coords[:, [x, y]] @ rot.T if cell_coords.size else None
        x_plot, y_plot = 0, 1
    else:
        x_plot, y_plot = x, y
        node_coords = node_coords[:, [x_plot, y_plot]]
        cell_coords = cell_coords[:, [x_plot, y_plot]] if cell_coords.size else None

    mst = monocle["mst"]
    edges = [(edge.source, edge.target) for edge in mst.es]
    mst_names = mst.vs["name"]
    branch_indices = []
    branch_labels = []
    for idx, name in enumerate(monocle.get("branch_points", [])):
        if name in mst_names:
            branch_indices.append(mst_names.index(name))
            branch_labels.append(str(idx + 1))

    return _GraphData(
        node_coords=node_coords,
        edges=edges,
        branch_indices=branch_indices,
        branch_labels=branch_labels,
        cell_coords=cell_coords,
    )


def _paga_graph_data(adata, *, basis="X_umap", groups=None, x=0, y=1):
    if "paga" not in adata.uns:
        raise KeyError("adata.uns['paga'] is missing.")
    paga = adata.uns["paga"]
    groups = groups or paga.get("groups")
    if groups is None or groups not in adata.obs:
        raise KeyError("PAGA trajectory plotting requires a valid `groups` key.")

    basis = _basis_key(adata, basis)
    cell_coords = np.asarray(adata.obsm[basis])[:, [x, y]]
    labels = pd.Series(adata.obs[groups].values, index=adata.obs_names)
    if isinstance(labels.dtype, pd.CategoricalDtype):
        categories = list(labels.dtype.categories)
    else:
        categories = list(pd.unique(labels))

    node_coords = []
    for category in categories:
        mask = labels == category
        if mask.any():
            node_coords.append(np.nanmedian(cell_coords[np.asarray(mask)], axis=0))
        else:
            node_coords.append([np.nan, np.nan])
    node_coords = np.asarray(node_coords, dtype=float)

    graph = (
        paga.get("connectivities_tree")
        if paga.get("connectivities_tree") is not None
        else paga.get("connectivities")
    )
    if graph is None:
        raise KeyError("adata.uns['paga'] has no connectivities graph.")
    graph = graph.tocoo() if sparse.issparse(graph) else sparse.coo_matrix(graph)
    edges = []
    edge_weights = []
    for i, j, value in zip(graph.row, graph.col, graph.data):
        if i < j and value > 0:
            edges.append((int(i), int(j)))
            edge_weights.append(float(value))
    degree = np.zeros(len(categories), dtype=int)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    branch_indices = [int(i) for i in np.where(degree > 2)[0]]
    branch_labels = [str(categories[i]) for i in branch_indices]

    directed_edges = None
    transitions = paga.get("transitions_confidence")
    if transitions is not None:
        transitions = (
            transitions.tocsr()
            if sparse.issparse(transitions)
            else sparse.csr_matrix(transitions)
        )
        directed_edges = []
        for i, j in edges:
            forward = float(transitions[i, j])
            backward = float(transitions[j, i])
            if forward <= 0 and backward <= 0:
                continue
            # scVelo stores PAGA transition confidence transposed for plotting.
            if backward > forward:
                directed_edges.append((i, j, backward))
            else:
                directed_edges.append((j, i, forward))

    return _GraphData(
        node_coords=node_coords,
        edges=edges,
        branch_indices=branch_indices,
        branch_labels=branch_labels,
        cell_coords=cell_coords,
        edge_weights=edge_weights,
        directed_edges=directed_edges,
    )


def _rotation_matrix(theta_degrees):
    theta = np.radians(theta_degrees)
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


def _resolve_graph_data(
    adata,
    *,
    method="auto",
    basis=None,
    groups=None,
    model=None,
    x=0,
    y=1,
    theta=0.0,
):
    method = _normalize_method(adata, method=method, model=model)
    basis = _default_basis(method) if basis is None else basis
    if method == "monocle":
        return _monocle_graph_data(adata, x=x, y=y, theta=theta), method
    if method == "paga":
        return _paga_graph_data(adata, basis=basis, groups=groups, x=x, y=y), method
    if method == "slingshot":
        if model is None or not hasattr(model, "curves"):
            raise ValueError("`method='slingshot'` requires a Slingshot model.")
        return None, method
    raise NotImplementedError(f"Unsupported trajectory method: {method!r}")


def _draw_graph(
    ax,
    graph_data,
    *,
    show_tree=True,
    show_branch_points=True,
    backbone_color="#8A8A8A",
    cell_link_size=1.15,
    branch_point_size=150,
    branch_point_color="#6F6F6F",
    branch_point_label_color="white",
    branch_point_label_fontsize=10,
    tree_alpha=0.85,
    zorder_base=3,
    arrow=False,
):
    if show_tree:
        if arrow and graph_data.directed_edges:
            for i, j in graph_data.edges:
                ax.plot(
                    [graph_data.node_coords[i, 0], graph_data.node_coords[j, 0]],
                    [graph_data.node_coords[i, 1], graph_data.node_coords[j, 1]],
                    color=backbone_color,
                    linewidth=max(cell_link_size * 0.65, 0.9),
                    alpha=min(tree_alpha, 0.45),
                    zorder=zorder_base,
                    solid_capstyle="round",
                )
            weights = np.asarray([edge[2] for edge in graph_data.directed_edges], dtype=float)
            finite = weights[np.isfinite(weights)]
            max_weight = float(np.nanmax(finite)) if len(finite) else 1.0
            max_weight = max(max_weight, 1e-12)
            for i, j, weight in graph_data.directed_edges:
                start = graph_data.node_coords[i]
                end = graph_data.node_coords[j]
                direction = end - start
                if np.linalg.norm(direction) < 1e-8:
                    continue
                start_px, end_px = ax.transData.transform([start, end])
                if np.linalg.norm(end_px - start_px) < 34.0:
                    continue
                linewidth = cell_link_size * (0.75 + 0.5 * weight / max_weight)
                patch = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    shrinkA=10,
                    shrinkB=10,
                    linewidth=linewidth,
                    color=backbone_color,
                    alpha=tree_alpha,
                    zorder=zorder_base,
                    connectionstyle="arc3,rad=0.0",
                )
                ax.add_patch(patch)
        else:
            for i, j in graph_data.edges:
                ax.plot(
                    [graph_data.node_coords[i, 0], graph_data.node_coords[j, 0]],
                    [graph_data.node_coords[i, 1], graph_data.node_coords[j, 1]],
                    color=backbone_color,
                    linewidth=cell_link_size,
                    alpha=tree_alpha,
                    zorder=zorder_base,
                )

    if show_branch_points and graph_data.branch_indices:
        anchors = graph_data.node_coords[graph_data.branch_indices]
        avoid_coords = (
            graph_data.cell_coords
            if graph_data.cell_coords is not None and len(graph_data.cell_coords) > 0
            else graph_data.node_coords
        )
        _draw_branch_points(
            ax,
            anchors,
            graph_data.branch_labels,
            size=branch_point_size,
            facecolor=branch_point_color,
            text_color=branch_point_label_color,
            fontsize=branch_point_label_fontsize,
            alpha=0.92,
            zorder=zorder_base + 2,
            avoid_coords=avoid_coords,
        )
    return ax


def _draw_slingshot_curves(
    ax,
    model,
    *,
    color="#8A8A8A",
    linewidth=1.15,
    alpha=0.85,
    zorder=3,
):
    for curve in getattr(model, "curves", []):
        points = getattr(curve, "points_interp", None)
        order = getattr(curve, "order", None)
        if points is None:
            continue
        points = np.asarray(points)
        if order is not None and len(order) == len(points):
            points = points[np.asarray(order)]
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    return ax


@register_function(
    aliases=[
        "trajectory_overlay",
        "trajectory overlay",
        "trajectory backbone overlay",
        "轨迹叠加",
        "轨迹骨架叠加",
    ],
    category="pl",
    description=(
        "Overlay a method-specific trajectory backbone or lineage curve on an "
        "existing embedding axis."
    ),
    examples=[
        (
            "fig, ax = plt.subplots(figsize=(4, 4)); "
            "ov.pl.embedding(adata, basis='X_umap', color='clusters', "
            "ax=ax, show=False); "
            "ov.pl.trajectory_overlay(adata, ax=ax, method='paga', "
            "groups='clusters')"
        ),
        (
            "ov.pl.trajectory_overlay(mono.adata, ax=ax, method='monocle', "
            "basis='X_DDRTree')"
        ),
    ],
    related=[
        "pl.trajectory",
        "pl.trajectory_tree",
        "pl.embedding",
        "utils.cal_paga",
    ],
)
def trajectory_overlay(
    adata,
    ax,
    *,
    method="auto",
    basis=None,
    groups=None,
    model=None,
    x=0,
    y=1,
    show_tree=True,
    show_branch_points=True,
    backbone_color="#8A8A8A",
    cell_link_size=1.25,
    branch_point_size=180,
    branch_point_color="#6F6F6F",
    branch_point_label_color="white",
    branch_point_label_fontsize=10,
    theta=0.0,
    zorder_base=3,
):
    """Overlay a trajectory backbone on an existing embedding axis.

    This function adds only the inferred trajectory structure to ``ax``. It is
    intended for composing with :func:`omicverse.pl.embedding` or other
    embedding plots where the cells, colors, legends, and titles are already
    controlled by the caller.

    Parameters
    ----------
    adata
        Annotated data matrix containing the trajectory result.
    ax
        Existing matplotlib axes on which the trajectory should be drawn.
    method
        Trajectory backend. ``"auto"`` selects Monocle when
        ``adata.uns['monocle']`` is available, then PAGA when
        ``adata.uns['paga']`` is available, and otherwise a supported model.
        Explicit values currently supported by the overlay are ``"monocle"``,
        ``"paga"``, and ``"slingshot"``.
    basis
        Low-dimensional representation used by graph-based methods. For
        Monocle this is usually ``"X_DDRTree"``; for PAGA this is commonly
        ``"X_umap"``. When ``None``, the default is selected from ``method``.
        The ``"umap"`` shorthand is also accepted when ``adata.obsm['X_umap']``
        exists.
    groups
        Observation key defining PAGA groups. When omitted, the key stored in
        ``adata.uns['paga']['groups']`` is used.
    model
        Method-specific model object. ``method='slingshot'`` expects a fitted
        Slingshot model exposing ``curves``.
    x, y
        Components in ``basis`` or Monocle DDRTree coordinates to plot.
    show_tree
        Whether to draw the trajectory graph or fitted lineage curves.
    show_branch_points
        Whether to draw branch point labels for graph methods.
    backbone_color
        Color used for trajectory backbone lines or curves.
    cell_link_size
        Line width of trajectory backbone edges.
    branch_point_size
        Marker size for branch point labels.
    branch_point_color
        Fill color for branch point markers.
    branch_point_label_color
        Text color for branch point labels.
    branch_point_label_fontsize
        Font size for branch point labels.
    theta
        Rotation angle in degrees for Monocle DDRTree coordinates.
    zorder_base
        Base z-order for trajectory layers.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the trajectory overlay added.
    """
    graph_data, method = _resolve_graph_data(
        adata,
        method=method,
        basis=basis,
        groups=groups,
        model=model,
        x=x,
        y=y,
        theta=theta,
    )
    if method == "slingshot":
        return _draw_slingshot_curves(
            ax,
            model,
            color=backbone_color,
            linewidth=cell_link_size,
            zorder=zorder_base,
        )

    use_arrow = method == "paga" and graph_data.directed_edges
    if method == "paga" and backbone_color == "#8A8A8A":
        backbone_color = "#1A1A1A"
    if method == "paga" and cell_link_size <= 1.25:
        cell_link_size = 1.8
    return _draw_graph(
        ax,
        graph_data,
        show_tree=show_tree,
        show_branch_points=show_branch_points,
        backbone_color=backbone_color,
        cell_link_size=cell_link_size,
        branch_point_size=branch_point_size,
        branch_point_color=branch_point_color,
        branch_point_label_color=branch_point_label_color,
        branch_point_label_fontsize=branch_point_label_fontsize,
        zorder_base=zorder_base,
        arrow=use_arrow,
    )


@register_function(
    aliases=[
        "trajectory",
        "trajectory plot",
        "trajectory graph",
        "轨迹图",
        "通用轨迹图",
    ],
    category="pl",
    description=(
        "Plot cells on an embedding and overlay a method-specific trajectory "
        "graph."
    ),
    examples=[
        (
            "ov.pl.trajectory(adata, method='paga', basis='X_umap', "
            "groups='clusters', color='clusters')"
        ),
        (
            "ov.pl.trajectory(mono.adata, method='monocle', "
            "basis='X_DDRTree', color='State')"
        ),
    ],
    related=[
        "pl.trajectory_overlay",
        "pl.trajectory_tree",
        "pl.embedding",
        "utils.cal_paga",
    ],
)
def trajectory(
    adata,
    *,
    basis=None,
    color=None,
    color_by=None,
    method="auto",
    groups=None,
    model=None,
    x=0,
    y=1,
    size=50,
    figsize=(5, 4),
    frameon="small",
    show_tree=True,
    show_branch_points=True,
    backbone_color="#8A8A8A",
    cell_link_size=1.15,
    branch_point_size=150,
    branch_point_color="#6F6F6F",
    branch_point_label_color="white",
    branch_point_label_fontsize=10,
    legend_loc="right margin",
    legend_fontsize=None,
    cmap=None,
    title=None,
    ax=None,
    show=None,
    save=None,
    dpi=150,
    **embedding_kwargs,
):
    """Draw cells in an embedding and overlay a trajectory backbone.

    This is the high-level trajectory view. It first draws cells with
    :func:`omicverse.pl.embedding`, then calls :func:`trajectory_overlay` to add
    the inferred graph or lineage curves. Use :func:`trajectory_overlay`
    directly when you need to compose the trajectory with a pre-existing axes.

    Parameters
    ----------
    adata
        Annotated data matrix containing the embedding and trajectory result.
    basis
        Low-dimensional representation used for the cell scatter. When
        ``None``, Monocle uses ``"X_DDRTree"`` and other supported methods use
        ``"X_umap"``.
    color
        Observation or variable key passed to :func:`omicverse.pl.embedding` for
        coloring cells. If omitted, ``"State"`` is used when present.
    color_by
        Backward-compatible alias for ``color``.
    method
        Trajectory backend. ``"auto"`` selects Monocle, then PAGA, then a
        supported model. Explicit values currently supported are
        ``"monocle"``, ``"paga"``, and ``"slingshot"``.
    groups
        Observation key defining PAGA groups. When omitted, the key stored in
        ``adata.uns['paga']['groups']`` is used.
    model
        Method-specific model object, currently used for Slingshot curves.
    x, y
        Components in ``basis`` or Monocle DDRTree coordinates to plot.
    size
        Cell marker size passed to :func:`omicverse.pl.embedding`.
    figsize
        Figure size used when ``ax`` is not provided.
    frameon
        Frame style passed to :func:`omicverse.pl.embedding`.
    show_tree
        Whether to draw the trajectory graph or fitted lineage curves.
    show_branch_points
        Whether to draw branch point labels for graph methods.
    backbone_color
        Color used for trajectory backbone lines or curves.
    cell_link_size
        Line width of trajectory backbone edges.
    branch_point_size
        Marker size for branch point labels.
    branch_point_color
        Fill color for branch point markers.
    branch_point_label_color
        Text color for branch point labels.
    branch_point_label_fontsize
        Font size for branch point labels.
    legend_loc
        Legend location passed to :func:`omicverse.pl.embedding`.
    legend_fontsize
        Legend font size passed to :func:`omicverse.pl.embedding`.
    cmap
        Colormap used for continuous cell colors.
    title
        Plot title. When omitted, the color key is used.
    ax
        Existing matplotlib axes. When omitted, a new figure is created.
    show
        Whether to call :func:`matplotlib.pyplot.show`.
    save
        Optional output path passed to ``Figure.savefig``.
    dpi
        Resolution used when saving the figure.
    **embedding_kwargs
        Additional keyword arguments forwarded to
        :func:`omicverse.pl.embedding`.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes containing the embedding and trajectory overlay.
    """
    resolved_method = _normalize_method(adata, method=method, model=model)
    basis = _default_basis(resolved_method) if basis is None else basis
    color = color if color is not None else color_by
    if color is None:
        color = "State" if "State" in adata.obs else None
    if resolved_method == "paga" and tuple(figsize) == (5, 4):
        figsize = (4, 4)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    embedding(
        adata,
        basis=basis,
        color=color,
        ax=ax,
        size=size,
        frameon=frameon,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        cmap=cmap,
        title=title if title is not None else color,
        show=False,
        **embedding_kwargs,
    )
    trajectory_overlay(
        adata,
        ax,
        method=resolved_method,
        basis=basis,
        groups=groups,
        model=model,
        x=x,
        y=y,
        show_tree=show_tree,
        show_branch_points=show_branch_points,
        backbone_color=backbone_color,
        cell_link_size=cell_link_size,
        branch_point_size=branch_point_size,
        branch_point_color=branch_point_color,
        branch_point_label_color=branch_point_label_color,
        branch_point_label_fontsize=branch_point_label_fontsize,
    )

    if not (resolved_method == "paga" and legend_loc in {"right margin", "right"}):
        fig.tight_layout(
            rect=(0, 0, 0.86, 1)
            if legend_loc in {"right margin", "right"}
            else None
        )
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def _complex_branch_point_anchors(monocle, coords_sorted):
    branch_points = monocle.get("branch_points", [])
    if not branch_points:
        return [], []

    mst = monocle.get("mst")
    closest_vertex = monocle.get("pr_graph_cell_proj_closest_vertex")
    if mst is None or closest_vertex is None:
        return [], []

    mst_names = mst.vs["name"]
    closest_vertex = np.asarray(closest_vertex).ravel().astype(int)
    source_coords = monocle.get("reducedDimS")
    center_coords = monocle.get("reducedDimK")

    anchors = []
    labels = []
    for bp_idx, bp_name in enumerate(branch_points):
        if bp_name not in mst_names:
            continue
        bp_vertex = mst_names.index(bp_name)
        cell_idx = np.where(closest_vertex == bp_vertex)[0]
        if len(cell_idx) > 0:
            anchor = np.nanmedian(coords_sorted[cell_idx, :], axis=0)
        elif source_coords is not None and center_coords is not None:
            center = np.asarray(center_coords)[:, bp_vertex]
            distances = np.linalg.norm(np.asarray(source_coords).T - center, axis=1)
            anchor = coords_sorted[int(np.nanargmin(distances))]
        else:
            continue
        anchors.append(anchor)
        labels.append(bp_idx + 1)

    return anchors, labels


def _draw_tree_cells(
    ax,
    adata,
    coords_scatter,
    *,
    color,
    cell_size,
    cell_alpha,
    cmap,
    legend_loc,
    legend_fontsize,
):
    if color in adata.obs.columns:
        values = adata.obs[color].values
        if _is_categorical(values):
            color_map = _get_obs_color_map(adata, color, values)
            colors = [color_map[value] for value in values]
            ax.scatter(
                coords_scatter[:, 0],
                coords_scatter[:, 1],
                c=colors,
                s=cell_size ** 2 * 10,
                edgecolors="none",
                alpha=cell_alpha,
                zorder=2,
            )
            _build_categorical_legend(
                ax,
                color_map,
                title=color,
                anchor=legend_loc,
                fontsize=legend_fontsize,
            )
        else:
            scatter = ax.scatter(
                coords_scatter[:, 0],
                coords_scatter[:, 1],
                c=pd.to_numeric(adata.obs[color], errors="coerce").to_numpy(dtype=float),
                s=cell_size ** 2 * 10,
                cmap=cmap or "RdBu_r",
                zorder=2,
                edgecolors="none",
                alpha=cell_alpha,
            )
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label(color)
    else:
        ax.scatter(
            coords_scatter[:, 0],
            coords_scatter[:, 1],
            s=cell_size ** 2 * 10,
            edgecolors="none",
            alpha=cell_alpha,
            zorder=2,
        )


def _paga_tree_layout(
    adata,
    *,
    basis,
    groups,
    pseudotime,
    x,
    y,
    jitter_width,
):
    graph_data = _paga_graph_data(adata, basis=basis, groups=groups, x=x, y=y)
    paga = adata.uns["paga"]
    groups = groups or paga.get("groups")
    labels = pd.Series(adata.obs[groups].values, index=adata.obs_names)
    if isinstance(labels.dtype, pd.CategoricalDtype):
        categories = list(labels.dtype.categories)
    else:
        categories = list(pd.unique(labels))

    pseudotime_values = pd.to_numeric(
        adata.obs[pseudotime], errors="coerce"
    ).to_numpy(dtype=float)
    basis_cell_coords = graph_data.cell_coords.copy()
    basis_node_coords = graph_data.node_coords.copy()
    node_coords = basis_node_coords.copy()
    for idx, category in enumerate(categories):
        mask = np.asarray(labels == category)
        if mask.any():
            node_coords[idx, 1] = np.nanmedian(pseudotime_values[mask])
        else:
            node_coords[idx, 1] = np.nan

    n_nodes = len(categories)
    adj = [[] for _ in range(n_nodes)]
    for i, j in graph_data.edges:
        adj[i].append(j)
        adj[j].append(i)

    finite_times = np.where(np.isfinite(node_coords[:, 1]), node_coords[:, 1], np.inf)
    root_order = list(np.argsort(finite_times))
    parent = np.full(n_nodes, -1, dtype=int)
    children = [[] for _ in range(n_nodes)]
    visited = np.zeros(n_nodes, dtype=bool)
    for start in root_order:
        if visited[start]:
            continue
        queue = [int(start)]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            neighbors = [neighbor for neighbor in adj[node] if not visited[neighbor]]
            neighbors = sorted(
                neighbors,
                key=lambda idx: (
                    finite_times[idx],
                    graph_data.node_coords[idx, 0],
                    graph_data.node_coords[idx, 1],
                ),
            )
            for neighbor in neighbors:
                parent[neighbor] = node
                children[node].append(neighbor)
                visited[neighbor] = True
                queue.append(neighbor)

    x_positions = np.full(n_nodes, np.nan, dtype=float)
    next_leaf = 0

    def assign_x(node):
        nonlocal next_leaf
        if not children[node]:
            x_positions[node] = float(next_leaf)
            next_leaf += 1
        else:
            for child in children[node]:
                assign_x(child)
            x_positions[node] = float(np.nanmean(x_positions[children[node]]))

    for start in root_order:
        if np.isnan(x_positions[start]):
            assign_x(int(start))
            next_leaf += 1

    if np.all(np.isfinite(x_positions)):
        x_positions -= np.nanmean(x_positions)
        x_positions *= 1.65
        basis_x = basis_node_coords[:, 0].astype(float)
        basis_span = np.nanmax(basis_x) - np.nanmin(basis_x)
        tree_span = np.nanmax(x_positions) - np.nanmin(x_positions)
        if basis_span > 1e-8 and tree_span > 1e-8:
            basis_scaled = (basis_x - np.nanmean(basis_x)) / basis_span * tree_span
            corr = np.corrcoef(
                x_positions[np.isfinite(x_positions) & np.isfinite(basis_scaled)],
                basis_scaled[np.isfinite(x_positions) & np.isfinite(basis_scaled)],
            )[0, 1]
            if np.isfinite(corr) and corr < 0:
                basis_scaled *= -1
            x_positions = 0.64 * x_positions + 0.36 * basis_scaled
    else:
        x_positions = graph_data.node_coords[:, 0].copy()
    node_coords[:, 0] = x_positions

    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    cell_x = np.zeros(adata.n_obs, dtype=float)
    dx = np.nanmax(node_coords[:, 0]) - np.nanmin(node_coords[:, 0])
    dx = dx if dx > 0 else 1.0
    lane_width = max(0.34, min(0.85, dx * 0.22))
    for obs_idx, value in enumerate(labels):
        node = category_to_idx.get(value)
        if node is None:
            cell_x[obs_idx] = np.nan
            continue
        par = parent[node]
        if par >= 0 and np.isfinite(node_coords[par, 1]) and np.isfinite(node_coords[node, 1]):
            start_time = node_coords[par, 1]
            end_time = node_coords[node, 1]
            span = end_time - start_time
            if abs(span) > 1e-8:
                frac = np.clip((pseudotime_values[obs_idx] - start_time) / span, 0.0, 1.0)
                cell_x[obs_idx] = node_coords[par, 0] + frac * (
                    node_coords[node, 0] - node_coords[par, 0]
                )
            else:
                cell_x[obs_idx] = node_coords[node, 0]
        else:
            cell_x[obs_idx] = node_coords[node, 0]

    for category in categories:
        node = category_to_idx[category]
        mask = np.asarray(labels == category)
        if not mask.any():
            continue
        par = parent[node]
        if par >= 0:
            direction = basis_node_coords[node] - basis_node_coords[par]
        elif children[node]:
            direction = basis_node_coords[children[node][0]] - basis_node_coords[node]
        else:
            direction = np.array([1.0, 0.0])
        if np.linalg.norm(direction) > 1e-8:
            direction = direction / np.linalg.norm(direction)
            axis = np.array([-direction[1], direction[0]])
        else:
            axis = np.array([1.0, 0.0])
        center = np.nanmedian(basis_cell_coords[mask], axis=0)
        score = (basis_cell_coords[mask] - center) @ axis
        finite_score = score[np.isfinite(score)]
        scale = (
            np.nanpercentile(np.abs(finite_score), 90)
            if len(finite_score)
            else 0.0
        )
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = np.nanstd(finite_score) if len(finite_score) else 0.0
        if not np.isfinite(scale) or scale <= 1e-8:
            continue
        lateral = np.clip(score / scale, -1.0, 1.0) * lane_width
        cell_x[mask] += lateral

    coords_scatter = np.column_stack([cell_x, pseudotime_values])
    rng = np.random.default_rng(0)
    coords_scatter[:, 0] += rng.normal(
        0.0,
        max(dx * jitter_width, 0.026),
        size=coords_scatter.shape[0],
    )

    graph_data.node_coords = node_coords
    directed_edges = []
    edge_weights = graph_data.edge_weights or [1.0] * len(graph_data.edges)
    for (i, j), weight in zip(graph_data.edges, edge_weights):
        time_i = node_coords[i, 1]
        time_j = node_coords[j, 1]
        if np.isfinite(time_i) and np.isfinite(time_j) and time_i <= time_j:
            directed_edges.append((i, j, float(weight)))
        elif np.isfinite(time_i) and np.isfinite(time_j):
            directed_edges.append((j, i, float(weight)))
        elif parent[j] == i:
            directed_edges.append((i, j, float(weight)))
        elif parent[i] == j:
            directed_edges.append((j, i, float(weight)))
    graph_data.directed_edges = directed_edges
    return graph_data, coords_scatter


@register_function(
    aliases=[
        "trajectory_tree",
        "trajectory tree",
        "pseudotime tree",
        "轨迹树",
        "伪时间树图",
    ],
    category="pl",
    description=(
        "Draw a tree-layout trajectory view with real pseudotime on the "
        "y-axis."
    ),
    examples=[
        (
            "ov.pl.trajectory_tree(mono.adata, method='monocle', "
            "pseudotime='Pseudotime', color='State')"
        ),
        (
            "ov.pl.trajectory_tree(adata, method='paga', basis='X_umap', "
            "groups='clusters', pseudotime='dpt_pseudotime', "
            "color='clusters')"
        ),
    ],
    related=[
        "pl.trajectory",
        "pl.trajectory_overlay",
        "pl.branch_streamplot",
    ],
)
def trajectory_tree(
    adata,
    *,
    color=None,
    color_by=None,
    method="auto",
    basis="X_umap",
    groups=None,
    pseudotime="Pseudotime",
    show_branch_points=True,
    cell_size=2.0,
    cell_link_size=0.7,
    figsize=(5, 4),
    cmap=None,
    ax=None,
    legend_loc="right margin",
    legend_fontsize=None,
    cell_alpha=0.88,
    jitter_width=0.006,
    save=None,
    dpi=150,
    show=None,
):
    """Draw a tree-layout trajectory view using real pseudotime on the y-axis.

    The tree view separates branches horizontally while preserving the real
    pseudotime values on the vertical axis. The y-axis is inverted so smaller
    pseudotime values are shown at the top. Monocle uses the cell projection
    MST, while PAGA uses a tree-like layout derived from the PAGA graph and
    draws directed arrows from lower to higher pseudotime.

    Parameters
    ----------
    adata
        Annotated data matrix containing trajectory and pseudotime results.
    color
        Observation key used for coloring cells. If omitted, ``"State"`` is
        used when present.
    color_by
        Backward-compatible alias for ``color``.
    method
        Trajectory backend. ``trajectory_tree`` currently supports
        ``"monocle"`` and ``"paga"``.
    basis
        Low-dimensional representation used to initialize PAGA branch
        separation. Monocle tree layout does not use this parameter.
    groups
        Observation key defining PAGA groups. When omitted, the key stored in
        ``adata.uns['paga']['groups']`` is used.
    pseudotime
        Key in ``adata.obs`` containing pseudotime values for the y-axis.
    show_branch_points
        Whether to label branch nodes when branch points can be inferred.
    cell_size
        Base cell marker size.
    cell_link_size
        Line width for tree edges.
    figsize
        Figure size used when ``ax`` is not provided.
    cmap
        Colormap used for continuous cell colors.
    ax
        Existing matplotlib axes. When omitted, a new figure is created.
    legend_loc
        Legend location for categorical cell colors.
    legend_fontsize
        Legend font size for categorical cell colors.
    cell_alpha
        Cell marker opacity.
    jitter_width
        Horizontal jitter applied to cells for readability.
    save
        Optional output path passed to ``Figure.savefig``.
    dpi
        Resolution used when saving the figure.
    show
        Whether to call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes containing the pseudotime tree view.
    """
    method = _normalize_method(adata, method=method)
    if pseudotime not in adata.obs:
        raise KeyError(f"`{pseudotime}` was not found in `adata.obs`.")

    color = color if color is not None else color_by
    if color is None:
        color = "State" if "State" in adata.obs else None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    tree_cell_size = cell_size
    tree_cell_alpha = cell_alpha
    if method == "paga":
        if cell_size == 2.0:
            tree_cell_size = 1.0
        if cell_alpha == 0.88:
            tree_cell_alpha = 0.52

    branch_anchors = None
    branch_labels = None
    branch_avoid_coords = None
    paga_graph_data = None
    if method == "monocle":
        monocle = adata.uns["monocle"]
        if "pr_graph_cell_proj_tree" not in monocle:
            raise ValueError("Run order_cells() first to build the cell projection MST.")

        cell_mst = monocle["pr_graph_cell_proj_tree"]
        vertex_names = cell_mst.vs["name"]
        pseudotime_values = adata.obs.loc[vertex_names, pseudotime].to_numpy(dtype=float)
        root_idx = int(np.nanargmin(pseudotime_values))
        coords = np.asarray(cell_mst.layout_reingold_tilford(root=[root_idx]).coords)
        coords[:, 1] = pseudotime_values

        name_to_idx = {name: idx for idx, name in enumerate(vertex_names)}
        order = [name_to_idx[name] for name in adata.obs_names]
        coords_sorted = coords[order]
        dx = np.nanmax(coords[:, 0]) - np.nanmin(coords[:, 0])
        dx = dx if dx > 0 else 1.0
        rng = np.random.default_rng(0)
        coords_scatter = coords_sorted.copy()
        coords_scatter[:, 0] += rng.normal(
            0.0, dx * jitter_width, size=coords_scatter.shape[0]
        )

        for edge in cell_mst.es:
            i, j = edge.source, edge.target
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color="#8A8A8A",
                linewidth=cell_link_size,
                alpha=0.68,
                zorder=1,
            )

        if show_branch_points:
            branch_anchors, branch_labels = _complex_branch_point_anchors(
                monocle, coords_sorted
            )
            branch_avoid_coords = coords_scatter
    elif method == "paga":
        paga_graph_data, coords_scatter = _paga_tree_layout(
            adata,
            basis=basis,
            groups=groups,
            pseudotime=pseudotime,
            x=0,
            y=1,
            jitter_width=jitter_width,
        )
        if show_branch_points and paga_graph_data.branch_indices:
            branch_anchors = paga_graph_data.node_coords[paga_graph_data.branch_indices]
            branch_labels = paga_graph_data.branch_labels
            branch_avoid_coords = coords_scatter
    else:
        raise NotImplementedError(
            f"`trajectory_tree` does not support method={method!r}."
        )

    _draw_tree_cells(
        ax,
        adata,
        coords_scatter,
        color=color,
        cell_size=tree_cell_size,
        cell_alpha=tree_cell_alpha,
        cmap=cmap,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
    )
    if paga_graph_data is not None and not ax.yaxis_inverted():
        ax.invert_yaxis()
    if paga_graph_data is not None:
        _draw_graph(
            ax,
            paga_graph_data,
            show_tree=True,
            show_branch_points=False,
            backbone_color="#1A1A1A",
            cell_link_size=max(cell_link_size, 1.2),
            branch_point_size=150,
            branch_point_color="#6F6F6F",
            branch_point_label_color="white",
            branch_point_label_fontsize=10,
            tree_alpha=0.82,
            zorder_base=3,
            arrow=True,
        )
    if branch_anchors is not None:
        _draw_branch_points(
            ax,
            branch_anchors,
            branch_labels,
            size=150,
            facecolor="#6F6F6F",
            fontsize=10,
            alpha=0.92,
            zorder=5,
            avoid_coords=branch_avoid_coords,
        )

    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_ylabel(pseudotime)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", bottom=False)
    fig.tight_layout(rect=(0, 0, 0.86, 1) if legend_loc in {"right margin", "right"} else None)
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
