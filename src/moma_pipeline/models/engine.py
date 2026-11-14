"""Pipeline model manager - thin facade over omnishim's MoMa-compatible engine.

``MomaModelEngine`` delegates to ``omnishim.compat.moma.MoMaCellposeEngine``
(which reproduces the old MoMa_CPM model-management behavior: engine dispatch,
eval-param mapping, locks, log lines) and only adds package-level defaults:
direct model paths (decision D3) and no import-time GPU probing.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .defaults import DEFAULT_CELL_MODEL_PATH, DEFAULT_CHANNEL_MODEL_PATH


def gpu_available() -> bool:
    """True when torch reports a CUDA device (torch imported lazily)."""
    import torch
    return bool(torch.cuda.is_available())


def cellpose_version() -> str:
    """Installed cellpose version string ('unknown' when unavailable)."""
    try:
        import cellpose
        return str(getattr(cellpose, 'version', 'unknown'))
    except Exception:
        return 'unknown'


class MomaModelEngine:
    """Two-model (channel + cell) manager with the old MoMa_CPM duck-typed surface.

    Parameters
    ----------
    cell_model_path, channel_model_path:
        Checkpoint paths. None selects the package defaults from
        ``moma_pipeline.models.defaults`` (direct paths, decision D3).
    cell_model_params, channel_model_params:
        Eval-kwarg dicts. cell_model_params=None -> omnishim's MoMa cell
        defaults. channel_model_params=None -> the parameters matched to the
        channel checkpoint: diameter 30 / cellprob 0 / flow 0.4 / min_size 15
        for the Fine-tuned Deep channel family, and the unchanged historical
        MoMa channel defaults (diameter 60, min_size 1000) for every other
        checkpoint. An explicit dict is used as-is and always wins.
    cell_model_engine:
        'auto' (default), 'cellpose', 'omnishim' ('omnipose' alias).
    log_recorder:
        Optional object with write_log(str) (e.g. core.logging.LogRecorder).
    gpu:
        Keyword-only. None (default) uses CUDA when available, resolved lazily.

    The wrapped engine exposes load_models/release_models/eval_images,
    describe_models, eval_cell_params/eval_channel_params, cell_model/channel_model.
    """

    def __init__(self, cell_model_path: Optional[str] = None,
                 channel_model_path: Optional[str] = None,
                 cell_model_params: Optional[Dict[str, Any]] = None,
                 channel_model_params: Optional[Dict[str, Any]] = None,
                 cell_model_engine: Any = "auto", log_recorder: Any = None,
                 *, gpu: Optional[bool] = None):
        from omnishim.compat.moma import DEFAULT_CELL_PARAMS, MoMaCellposeEngine
        self._engine = MoMaCellposeEngine(
            cell_model_path=cell_model_path or DEFAULT_CELL_MODEL_PATH,
            channel_model_path=channel_model_path or DEFAULT_CHANNEL_MODEL_PATH,
            cell_model_params=(dict(DEFAULT_CELL_PARAMS) if cell_model_params is None
                               else cell_model_params),
            # None is passed through: omnishim resolves the parameters matched
            # to the channel checkpoint (legacy models keep their old dict).
            channel_model_params=channel_model_params,
            cell_model_engine=cell_model_engine,
            log_recorder=log_recorder, gpu=gpu)

    # -- duck-typed delegation (identical surface to the old MoMa_CPM) --------
    def load_models(self, model: str = "channel") -> None:
        return self._engine.load_models(model)

    def release_models(self, model: str = "all") -> None:
        return self._engine.release_models(model)

    def eval_images(self, imgs: Any, model: str = "channel") -> Any:
        return self._engine.eval_images(imgs, model=model)

    def describe_models(self) -> Dict[str, Any]:
        return self._engine.describe_models()

    @property
    def eval_cell_params(self) -> Dict[str, Any]:
        return self._engine.eval_cell_params

    @property
    def eval_channel_params(self) -> Dict[str, Any]:
        return self._engine.eval_channel_params

    @eval_channel_params.setter
    def eval_channel_params(self, value: Dict[str, Any]) -> None:
        self._engine.eval_channel_params = value

    @property
    def channel_params_source(self) -> str:
        """Where the channel eval parameters came from ('family:<f>' / 'legacy')."""
        return self._engine.channel_params_source

    @property
    def cell_model_engine(self) -> Any:
        return self._engine.cell_model_engine

    @property
    def cell_model_engine_resolved(self) -> Any:
        return self._engine.cell_model_engine_resolved

    @property
    def cell_model(self) -> Any:
        return self._engine.cell_model

    @property
    def channel_model(self) -> Any:
        return self._engine.channel_model

    @property
    def cell_model_path(self) -> Any:
        return self._engine.cell_model_path

    @property
    def channel_model_path(self) -> Any:
        return self._engine.channel_model_path

    def __repr__(self) -> str:
        return f"MomaModelEngine({self._engine!r})"


def create_engine(**kwargs) -> MomaModelEngine:
    """Create the pipeline model engine (see MomaModelEngine)."""
    return MomaModelEngine(**kwargs)
