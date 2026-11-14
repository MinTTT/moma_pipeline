"""Phase-2 integration verification (imports + CLI surfaces, no model run)."""
import subprocess, sys
from pathlib import Path

scripts = Path(sys.prefix) / "Scripts"

# 1) full package surface imports without heavy deps
import moma_pipeline.cli.fov_pipeline as fov_cli
import moma_pipeline.cli.nd2_pipeline as nd2_cli
import moma_pipeline.detection.pipeline as pipe
import moma_pipeline.detection.fov as det_fov
import moma_pipeline.detection.channels as chans
import moma_pipeline.detection.dt_utils as dtu
import moma_pipeline.io.fov_pkl as fov_pkl
assert "torch" not in sys.modules and "cellpose" not in sys.modules, \
    "pipeline imports must stay light until engine creation"
print("1) cli+detection modules import, no torch/cellpose leak: OK")

# 2) fov CLI --help shows the full original flag surface
p = subprocess.run([str(scripts / "moma-fov-pipeline.exe"), "--help"],
                   capture_output=True, text=True)
h = p.stdout
for flag in ("--fovs_dir", "--cell_detection_channel", "--threads", "--max_cells",
             "--cell_model_path", "--channel_model_path", "--cell_niter",
             "--skip_existing", "--dt_jobs", "--keep_models_loaded",
             "--flu_distribution", "--flu_distribution_channels",
             "--flu_distribution_bins"):
    assert flag in h, f"missing {flag} in fov --help"
print("2) moma-fov-pipeline --help has all DT flags: OK")

# 3) nd2 CLI --help
p = subprocess.run([str(scripts / "moma-nd2-pipeline.exe"), "--help"],
                   capture_output=True, text=True)
h = p.stdout
for flag in ("--nd2_path", "--save_dir", "--extract", "--extract_only", "--skip_extracted"):
    assert flag in h, f"missing {flag} in nd2 --help"
assert "auto" in h and "force" in h and "skip" in h
print("3) moma-nd2-pipeline --help has extraction flags: OK")

# 4) contract pieces exist with right names
assert callable(pipe.run_pipeline) and callable(pipe.add_dt_pipeline_args)
assert callable(pipe.build_run_parameters) and callable(pipe.dt_pipeline_kwargs_from_args)
assert det_fov.MomaFOV is not None and chans.resolve_channel_name_in_fov is not None
assert callable(dtu.resolve_dt_jobs) and callable(dtu.calculate_and_export_dt)
assert callable(fov_pkl.export_Moma_fov_data_dt) and callable(fov_pkl.load_Moma_fov_data_dt)
assert callable(fov_pkl.save_flu_distribution_dt)
print("4) detection/pipeline/fov_pkl contract names present: OK")

print("PHASE-2 INTEGRATION SMOKE OK")
