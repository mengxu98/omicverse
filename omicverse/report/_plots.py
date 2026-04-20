"""Render the ``viz`` specs that each provenance entry declares.

The report system is a dumb executor: it takes ``{"function":
"ov.pl.<fn>", "kwargs": {...}}`` specs stored by the dispatcher, resolves
them against the installed ``omicverse.pl``, runs the call, and
base64-encodes the resulting figure. There is no hardcoded step-to-plot
mapping here — if a dispatcher didn't declare a viz, the report has
nothing to draw.

``omicverse.style()`` is applied once per process before the first draw.
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_STYLE_APPLIED = False


def _apply_style_once() -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    try:
        import omicverse as ov
        ov.style()
    except Exception:  # noqa: BLE001
        pass
    _STYLE_APPLIED = True


def _resolve(path: str):
    """Resolve ``"ov.pl.embedding"`` → the callable.

    Accepts any dotted path rooted at ``omicverse`` (``ov`` → ``omicverse``).
    """
    if not path:
        return None
    parts = path.split(".")
    if not parts:
        return None
    head = parts[0]
    if head in ("ov", "omicverse"):
        import omicverse as ov
        obj: object = ov
    else:
        return None
    for p in parts[1:]:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj if callable(obj) else None


def _encode(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight",
                dpi=150, pad_inches=0.25)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fig_from(result) -> Optional[plt.Figure]:
    """``ov.pl.*`` helpers variously return Axes, Figure, dicts, or None."""
    if result is None:
        return plt.gcf() if plt.get_fignums() else None
    if isinstance(result, plt.Figure):
        return result
    if isinstance(result, plt.Axes):
        return result.figure
    if isinstance(result, dict):
        first = next(iter(result.values()), None)
        if isinstance(first, plt.Axes):
            return first.figure
        if isinstance(first, plt.Figure):
            return first
    return plt.gcf() if plt.get_fignums() else None


def _render_one(viz_entry: dict, adata) -> Optional[str]:
    fn = _resolve(viz_entry.get("function", ""))
    if fn is None:
        return None
    kwargs = dict(viz_entry.get("kwargs", {}) or {})
    # Friendly defaults for anything that takes `show` — we never want
    # the backend to try to open a window.
    kwargs.setdefault("show", False)
    try:
        result = fn(adata, **kwargs)
    except TypeError:
        # Some plotting helpers don't accept `show` — retry without it.
        kwargs.pop("show", None)
        result = fn(adata, **kwargs)
    fig = _fig_from(result)
    if fig is None:
        return None
    return _encode(fig)


def render(step, adata) -> list[str]:
    """Return a list of base64-encoded PNGs, one per declared viz.

    Any individual viz that blows up is swallowed — we would rather
    render a report with a missing figure than have one buggy viz spec
    kill the whole thing.
    """
    _apply_style_once()
    out: list[str] = []
    for v in getattr(step, "viz", None) or []:
        try:
            b64 = _render_one(v, adata)
        except Exception:  # noqa: BLE001
            b64 = None
        if b64:
            out.append(b64)
    return out
