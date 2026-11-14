"""Phase-2 models facade verification (does pull torch/cellpose by design)."""
import os, sys

import moma_pipeline.models as models
assert "torch" not in sys.modules, "torch must NOT be imported by moma_pipeline.models"
assert "cellpose" not in sys.modules, "cellpose must NOT be imported by moma_pipeline.models"
assert "omnishim" not in sys.modules, "omnishim must NOT be imported by moma_pipeline.models"
print("1) lazy: moma_pipeline.models imports without torch/cellpose/omnishim: OK")

from moma_pipeline.models import (DEFAULT_CELL_MODEL_PATH, DEFAULT_CHANNEL_MODEL_PATH,
                                  gpu_available)
print("2) defaults:", DEFAULT_CELL_MODEL_PATH)
print("   defaults:", DEFAULT_CHANNEL_MODEL_PATH)
assert os.path.exists(DEFAULT_CELL_MODEL_PATH), "default cell model missing!"
assert os.path.exists(DEFAULT_CHANNEL_MODEL_PATH), "default channel model missing!"
print("   both default checkpoints exist on disk: OK")

print("3) gpu_available():", gpu_available())
assert "torch" in sys.modules

from moma_pipeline.models import create_engine, MomaModelEngine, SingleModelSegmenter
eng = create_engine()
assert isinstance(eng, MomaModelEngine)
d = eng.describe_models()
print("4) engine describe_models(): engine resolved =", d["cell_model_engine_resolved"],
      "| nout =", d["cell_model_nout"], "| tier =", d["cell_model_tier"])
assert d["cell_model_engine_resolved"] in ("cellpose", "omnishim", "native")
assert d["channel_model_path"] and os.path.exists(d["channel_model_path"])
print("   eval_cell_params:", eng.eval_cell_params)
print("   eval_channel_params:", eng.eval_channel_params)

s = SingleModelSegmenter(DEFAULT_CHANNEL_MODEL_PATH)
print("5) segmenter resolved:", s.engine_resolved, "| checkpoint:", s.checkpoint_path)
assert s.engine_resolved in ("cellpose", "omnishim", "native")
print("   describe():", s.describe())

print("MODELS SMOKE OK")
