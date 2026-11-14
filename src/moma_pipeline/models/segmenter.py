"""Single-checkpoint segmenter (viewer use-case).

Replaces the raw ``cellpose.models.CellposeModel(gpu=True)`` usage of the old
``Moma_CPM_FOV_viewer``: one model (path or catalog name), engine dispatch via
omnishim, lazy load, eval, release.
"""
from __future__ import annotations

from typing import Any, Optional


def _empty_cuda_cache() -> None:
    import sys
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class SingleModelSegmenter:
    """Load one checkpoint and run eval on demand.

    Parameters
    ----------
    model_ref:
        Checkpoint path or omnishim catalog model name.
    engine:
        'auto' (default), 'cellpose', 'omnishim' ('omnipose' alias) - dispatch
        rules identical to the pipeline engine (omnishim.compat.resolve_engine).
    gpu:
        None (default) uses CUDA when available, resolved lazily.
    log_recorder:
        Optional object with write_log(str).
    """

    def __init__(self, model_ref: Any, *, engine: Any = "auto",
                 gpu: Optional[bool] = None, log_recorder: Any = None):
        self.model_ref = model_ref
        self.engine_requested = engine
        self.gpu = gpu
        self.log_recorder = log_recorder
        self._segmenter: Any = None
        self._engine_resolved: Optional[str] = None
        self._info: Any = None

    # -- resolution ------------------------------------------------------
    def _resolve(self):
        if self._info is None:
            from omnishim.compat import resolve_engine
            engine, info = resolve_engine(self.model_ref, self.engine_requested)
            self._engine_resolved = engine
            self._info = info
        return self._engine_resolved, self._info

    @property
    def engine_resolved(self) -> str:
        return self._resolve()[0]

    @property
    def info(self) -> Any:
        return self._resolve()[1]

    @property
    def checkpoint_path(self) -> str:
        return self._resolve()[1].path

    # -- lifecycle -------------------------------------------------------
    def _gpu_flag(self) -> bool:
        if self.gpu is not None:
            return bool(self.gpu)
        import torch
        return bool(torch.cuda.is_available())

    def load(self) -> "SingleModelSegmenter":
        """Load the model (no-op when already loaded)."""
        if self._segmenter is None:
            from omnishim import pipeline
            from omnishim.compat import ENGINE_DECODERS
            engine, info = self._resolve()
            kwargs: dict = {"decoder": ENGINE_DECODERS[engine],
                            "gpu": self._gpu_flag()}
            if engine == "omnishim":
                kwargs["nout"] = info.nout or 3
            self._segmenter = pipeline.Segmenter.from_checkpoint(
                info.path, **kwargs)
        return self

    @property
    def model(self) -> Any:
        """The underlying cellpose CellposeModel / omnishim OmniCellposeModel."""
        return self.load()._segmenter.model

    def eval(self, image: Any, **eval_params: Any) -> Any:
        """Evaluate one image; returns masks (first element of the eval result)."""
        model = self.model
        result = model.eval(image, **eval_params)
        # cellpose 4 returns (masks, flows, styles); omnishim likewise
        if isinstance(result, tuple):
            return result[0]
        return result

    def release(self) -> None:
        """Drop the model and empty the CUDA cache."""
        self._segmenter = None
        _empty_cuda_cache()

    def describe(self) -> dict:
        engine, info = self._resolve()
        return {
            "model_ref": str(self.model_ref),
            "checkpoint_path": info.path,
            "engine_requested": str(self.engine_requested),
            "engine_resolved": engine,
            "loaded": self._segmenter is not None,
        }

    @classmethod
    def from_name_or_path(cls, ref: Any, **kwargs) -> "SingleModelSegmenter":
        return cls(ref, **kwargs)

    def __repr__(self) -> str:
        try:
            return (f"SingleModelSegmenter({self.model_ref!r}, "
                    f"engine={self.engine_resolved!r})")
        except Exception:
            return f"SingleModelSegmenter({self.model_ref!r})"
