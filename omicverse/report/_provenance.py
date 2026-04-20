"""Lightweight per-step provenance recorded into ``adata.uns``.

Every tracked ``ov.*`` dispatcher appends a small dict to
``adata.uns['_ov_provenance']``. The dict is the single source of truth
for what the HTML report renders: function call, user-passed params,
backend label, timing, plus a *declarative* list of visualisations that
the dispatcher itself chose::

    {
        "name":         canonical step key (qc, pca, neighbors, ...)
        "function":     "ov.pp.qc"                # the public call path
        "params":       {kwarg: value, ...}       # ONLY what the user passed
        "backend":      "omicverse(cpu)" | "scrublet" | ...
        "mode":         ov.settings.mode at call time
        "timestamp":    ISO 8601
        "duration_s":   wall time (seconds)
        "version":      omicverse version
        "n_obs_before": # cells when the call started
        "n_obs_after":  # cells when the call returned
        "viz":          [                          # 0 or more figures
            {"function": "ov.pl.embedding",
             "kwargs":   {"basis": "X_umap", "color": "leiden"}},
            ...
        ],
    }

Provenance is the *only* thing the report reads. If a step has no
provenance entry, it did not run through an omicverse dispatcher — so it
doesn't appear in the report. No heuristics, no guessing.

The storage is a dict-of-dicts keyed by a zero-padded index
(``"000_qc"``, ``"001_umap"``, …) so it round-trips through h5ad
cleanly.
"""
from __future__ import annotations

import datetime
import functools
import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Optional, Union

# Thread-local depth counter. A dispatcher wraps its body in ``with tracked():``
# so that sub-calls to other tracked dispatchers see depth > 1 and their
# record_step() is a no-op. The report should reflect what the user
# actually typed, not every library-internal hop.
_state = threading.local()


def _depth() -> int:
    return getattr(_state, "depth", 0)


@contextmanager
def tracked():
    """Mark the enclosed block as a user-level tracked call.

    Nested invocations increment a thread-local counter; ``record_step``
    checks it and skips recording when depth > 1 (i.e. we're inside
    another tracked call). Exceptions still decrement the counter.
    """
    prev = _depth()
    _state.depth = prev + 1
    try:
        yield
    finally:
        _state.depth = prev


def tracks_depth(fn):
    """Function-decorator version of :func:`tracked`.

    Wraps a dispatcher so that every call bumps the thread-local depth
    counter for the duration of the body. Does *not* record — the
    dispatcher still calls :func:`record_step` inline at the right
    moment; the decorator only exists so that internal sub-calls to
    other tracked dispatchers see depth > 1 and skip their own record.
    """
    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        prev = _depth()
        _state.depth = prev + 1
        try:
            return fn(*args, **kwargs)
        finally:
            _state.depth = prev
    return _wrapper

PROVENANCE_KEY = "_ov_provenance"

_PRIMITIVE = (str, int, float, bool, type(None))


def _coerce(v: Any) -> Any:
    """Best-effort conversion to something h5ad/JSON can serialise."""
    if isinstance(v, _PRIMITIVE):
        return v
    if isinstance(v, (list, tuple)):
        return [_coerce(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _coerce(val) for k, val in v.items()}
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
    except ImportError:
        pass
    return repr(v)[:200]


def _omicverse_version() -> str:
    try:
        from omicverse import __version__ as v
        return str(v)
    except Exception:  # noqa: BLE001
        return ""


def _settings_mode() -> str:
    try:
        from omicverse._settings import settings
        return str(settings.mode)
    except Exception:  # noqa: BLE001
        return ""


def record_step(
    adata,
    name: str,
    *,
    function: str,
    params: Optional[dict] = None,
    backend: str = "",
    duration_s: Optional[float] = None,
    n_obs_before: Optional[int] = None,
    n_obs_after: Optional[int] = None,
    viz: Optional[list[dict]] = None,
) -> None:
    """Append a provenance entry to ``adata.uns['_ov_provenance']``.

    ``viz`` is a list of ``{"function": "ov.pl.<fn>", "kwargs": {...}}``
    dicts the caller wants the report to render. Every dispatcher knows
    best what to visualise for its own output, so it ships its viz spec
    alongside the call record — the report system is then a dumb
    executor, not a guesser.

    Best-effort: failures here never crash the user's pipeline.
    """
    # Skip when we're inside another tracked dispatcher — the outer one
    # is the user-visible call, so it owns the record.
    if _depth() > 1:
        return
    try:
        if adata is None or not hasattr(adata, "uns"):
            return
        entry = {
            "name": str(name),
            "function": str(function),
            "params": _coerce(params or {}),
            "backend": str(backend or ""),
            "mode": _settings_mode(),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "duration_s": float(duration_s) if duration_s is not None else None,
            "version": _omicverse_version(),
            "n_obs_before": int(n_obs_before) if n_obs_before is not None else None,
            "n_obs_after": int(n_obs_after) if n_obs_after is not None else int(adata.n_obs),
            # viz is list-of-dicts; h5ad can't store that so serialise to a
            # JSON string. get_provenance() decodes back.
            "viz": json.dumps(_coerce(viz or []), default=str),
        }
        log = dict(adata.uns.get(PROVENANCE_KEY, {}) or {})
        if isinstance(log, list):  # migrate legacy list-of-dicts
            log = {f"{i:03d}_{e.get('name', 'step')}": e for i, e in enumerate(log)}
        idx = f"{len(log):03d}_{name}"
        log[idx] = entry
        adata.uns[PROVENANCE_KEY] = log
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def track(adata, name: str, *, function: str, params: Optional[dict] = None,
          backend: str = "", viz: Optional[list[dict]] = None):
    """Context manager that times the wrapped block and records a step.

    ``viz`` may be either a literal list of dicts or a callable
    ``(adata) -> list[dict]`` — the callable form is resolved at exit
    time so you can reference state that only exists after the step ran
    (e.g. a cluster label that doesn't exist yet at entry).
    """
    n_before = getattr(adata, "n_obs", None)
    t0 = time.time()
    try:
        yield
    finally:
        if callable(viz):
            try:
                viz_resolved = viz(adata)
            except Exception:  # noqa: BLE001
                viz_resolved = []
        else:
            viz_resolved = viz
        record_step(
            adata, name,
            function=function, params=params, backend=backend,
            duration_s=time.time() - t0,
            n_obs_before=n_before,
            n_obs_after=getattr(adata, "n_obs", None),
            viz=viz_resolved,
        )


def _decode_entry(entry: dict) -> dict:
    """Decode JSON-encoded fields like ``viz`` back to Python objects."""
    if not isinstance(entry, dict):
        return entry
    out = dict(entry)
    viz = out.get("viz")
    if isinstance(viz, str):
        try:
            out["viz"] = json.loads(viz) if viz else []
        except Exception:  # noqa: BLE001
            out["viz"] = []
    return out


def get_provenance(adata) -> list[dict]:
    """Return the list of recorded steps (oldest first); empty if none."""
    if adata is None or not hasattr(adata, "uns"):
        return []
    raw = adata.uns.get(PROVENANCE_KEY, None)
    if not raw:
        return []
    if isinstance(raw, list):
        return [_decode_entry(e) for e in raw]
    if isinstance(raw, dict):
        return [_decode_entry(raw[k]) for k in sorted(raw.keys())]
    return []


def clear_provenance(adata) -> None:
    """Drop all provenance entries."""
    if adata is not None and hasattr(adata, "uns") and PROVENANCE_KEY in adata.uns:
        del adata.uns[PROVENANCE_KEY]


def tracked_step(
    name: str,
    function: str,
    *,
    backend: Union[str, Callable] = "omicverse",
    viz: Union[list, Callable] = (),
):
    """Decorator: record provenance when this dispatcher runs.

    Only captures keyword args the user actually passed — positional
    args beyond ``adata`` are not introspected. ``backend`` and ``viz``
    may be callables ``(adata, kwargs) -> value`` resolved at call time.

    Usage::

        @tracked_step(name='umap', function='ov.pp.umap',
                      backend=lambda a, kw: f'omicverse({_settings_mode()})',
                      viz=lambda a, kw: [{'function': 'ov.pl.embedding',
                                           'kwargs': {...}}])
        def umap(adata, **kwargs):
            ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            adata_in = args[0] if args else kwargs.get("adata")
            t0 = time.time()
            result = fn(*args, **kwargs)
            # `copy=True` dispatchers return a new adata — prefer that target.
            target = result if (result is not None and hasattr(result, "uns")) \
                              else adata_in
            try:
                backend_str = backend(target, kwargs) if callable(backend) else str(backend)
            except Exception:  # noqa: BLE001
                backend_str = str(backend) if not callable(backend) else "omicverse"
            try:
                viz_list = viz(target, kwargs) if callable(viz) else list(viz)
            except Exception:  # noqa: BLE001
                viz_list = []
            # Drop kwargs that typically just say "please return a copy" —
            # they don't describe the step, they describe the caller.
            recorded = {k: v for k, v in kwargs.items()
                        if k not in ("copy", "inplace")}
            record_step(target, name, function=function,
                         params=recorded, backend=backend_str,
                         duration_s=time.time() - t0, viz=viz_list)
            return result
        return wrapper
    return decorator


def pick_color_key(adata, preference: Optional[Iterable[str]] = None) -> Optional[str]:
    """Helper for dispatchers' viz specs: first existing obs column.

    Used when declaring an ``ov.pl.embedding`` viz whose ``color`` should
    default to the most informative labeling already on the AnnData.
    """
    candidates = list(preference or ("cell_type", "leiden", "louvain",
                                      "cluster", "phase"))
    for k in candidates:
        if k in adata.obs.columns:
            return k
    return None
