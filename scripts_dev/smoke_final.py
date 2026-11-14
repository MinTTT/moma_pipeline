"""Final verification: whole-package surface, leak scan, all 6 CLIs."""
import subprocess, sys
from pathlib import Path

scripts = Path(sys.prefix) / "Scripts"

# 1) every package module imports; heavy deps stay out until models/viewer use
import importlib
mods = [
    "moma_pipeline",
    "moma_pipeline.core.fov_utils", "moma_pipeline.core.logging",
    "moma_pipeline.io.images", "moma_pipeline.io.contours",
    "moma_pipeline.io.transactions", "moma_pipeline.io.nd2_extract",
    "moma_pipeline.io.dt_exports", "moma_pipeline.io.fov_pkl",
    "moma_pipeline.morphology.voronoi_dt", "moma_pipeline.morphology.kernels",
    "moma_pipeline.morphology.celldt", "moma_pipeline.morphology.worker",
    "moma_pipeline.models",
    "moma_pipeline.detection.channels", "moma_pipeline.detection.fov",
    "moma_pipeline.detection.dt_utils", "moma_pipeline.detection.pipeline",
    "moma_pipeline.cli.fov_pipeline", "moma_pipeline.cli.nd2_pipeline",
    "moma_pipeline.cli.cell_parameters", "moma_pipeline.cli.flu_distribution",
]
for m in mods:
    importlib.import_module(m)
assert "torch" not in sys.modules and "cellpose" not in sys.modules
print("1) all 22 core modules import, no torch/cellpose leak: OK")

# viewers import headless-safe (napari/Qt OK, no QApplication, no cellpose)
import moma_pipeline.viewers.lazy_timelapse as lt
import moma_pipeline.viewers.fov_viewer as fv
from qtpy.QtWidgets import QApplication
assert QApplication.instance() is None
assert "cellpose" not in sys.modules
print("2) viewers import headless-safe (no QApplication, no cellpose): OK")

# 3) all 6 console scripts respond to --help with their real surface
expect = {
    "moma-fov-pipeline": ["--fovs_dir", "--dt_jobs", "--flu_distribution"],
    "moma-nd2-pipeline": ["--nd2_path", "--extract", "--extract_only"],
    "moma-cell-parameters": ["--pkl_dir", "--dry_run", "--pole_refinement"],
    "moma-flu-distribution": ["--pkl_dir", "--channels", "--n-bins"],
    "moma-view-timelapse": ["root"],
    "moma-view-fov": ["exported_data_dir", "--model_dir"],
}
for cmd, flags in expect.items():
    p = subprocess.run([str(scripts / (cmd + ".exe")), "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0, (cmd, p.stderr[:400])
    for f in flags:
        assert f in p.stdout, f"{cmd} --help missing {f}"
print("3) all 6 console scripts --help surfaces: OK")

print("FINAL SMOKE OK")
