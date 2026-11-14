"""Model usage facade - the ONLY place that imports cellpose/torch/omnishim.

Names resolve lazily: importing moma_pipeline.models never pulls torch/cellpose.

- create_engine / MomaModelEngine: two-model pipeline manager (delegates to
  omnishim.compat.moma.MoMaCellposeEngine).
- SingleModelSegmenter: one-checkpoint service (viewer use-case).
- defaults: direct model paths (MOMA_MODEL_ROOT env override, decision D3).
"""
from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

_LAZY_ATTRS: Dict[str, str] = {
    "create_engine": ".engine",
    "MomaModelEngine": ".engine",
    "gpu_available": ".engine",
    "cellpose_version": ".engine",
    "SingleModelSegmenter": ".segmenter",
    "DEFAULT_CELL_MODEL_PATH": ".defaults",
    "DEFAULT_CHANNEL_MODEL_PATH": ".defaults",
    "DEFAULT_MODEL_ROOT": ".defaults",
    "LEGACY_CHANNEL_MODEL_PATH": ".defaults",
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        value = getattr(import_module(_LAZY_ATTRS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(globals()))
