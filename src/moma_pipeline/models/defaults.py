"""Package model defaults (decision D3: direct path assignment).

Model paths are assigned directly in this project; the omnishim model catalog
is only a fallback for name resolution and configuring it is omnishim's concern.
Override the root with the MOMA_MODEL_ROOT environment variable, or pass explicit
paths via --cell_model_path / --channel_model_path (or the engine constructor).

Channel model (changed 2026-11-13)
----------------------------------
The default channel model is the Fine-tuned Deep channel segmentation model
(dino_omnipose_channel_deep_r001_20261108_s20260904, SHA-256 ec4ec9e5...). Its
tensors declare the omnishim family "cellpose4_deep", so omnishim loads it with
the DeepRefiner encoder and applies its model-matched evaluation parameters
(diameter 30, cellprob_threshold 0, flow_threshold 0.4, min_size 15).

Rollback to the previous channel model::

    moma-fov-pipeline -f <fovs_dir> --channel_model_path <root>/cellpose_sideGrooves_20250708-2

(or set MOMA_MODEL_ROOT / edit DEFAULT_CHANNEL_MODEL_PATH). That checkpoint has
no refinement head, so it is loaded by the plain cellpose loader and evaluated
with the historical channel parameters (diameter 60, cellprob_threshold 0,
flow_threshold 0.4, min_size 1000) - i.e. the old behaviour, unchanged.
"""
from __future__ import annotations

import os

DEFAULT_MODEL_ROOT = os.environ.get("MOMA_MODEL_ROOT",
                                    r"G:\python_code\Moma_cellPose\CPModels")

# Current DT pipeline defaults (were ./CPModels/... CWD-relative in the old scripts).
DEFAULT_CELL_MODEL_PATH = os.path.join(DEFAULT_MODEL_ROOT, "models",
                                       "cellpose_cells_20260525")
#: Fine-tuned Deep channel model (omnishim family "cellpose4_deep"; SHA-256
#: ec4ec9e582cef7cc54fa1b479771973f916936708a09f6e62435a6cce2d36b89). omnishim
#: loads it with its DeepRefiner encoder and evaluates it with its model-matched
#: parameters (diameter 30, cellprob_threshold 0, flow_threshold 0.4, min_size 15).
#: Rollback: --channel_model_path <root>/cellpose_sideGrooves_20250708-2 restores
#: the previous model, loader and parameters (diameter 60, min_size 1000).
DEFAULT_CHANNEL_MODEL_PATH = os.path.join(
    DEFAULT_MODEL_ROOT, "models",
    "dino_omnipose_channel_deep_r001_20261108_s20260904")
#: Previous default channel model (rollback target).
LEGACY_CHANNEL_MODEL_PATH = os.path.join(DEFAULT_MODEL_ROOT,
                                         "cellpose_sideGrooves_20250708-2")
