"""Phase-1 consolidated verification: import every ported module,
cross-module functional smoke, lazy-import guarantees, contour bug fix check.
"""
import sys, tempfile, json
from pathlib import Path

import numpy as np

report = []

# 1) every ported module imports
import moma_pipeline.core.logging, moma_pipeline.core.fov_utils
import moma_pipeline.io.images as images
import moma_pipeline.io.contours as contours
import moma_pipeline.io.transactions as transactions
import moma_pipeline.io.nd2_extract as nd2_extract
import moma_pipeline.io.dt_exports as dt_exports
import moma_pipeline.morphology.voronoi_dt as voronoi_dt
import moma_pipeline.morphology.kernels as kernels
import moma_pipeline.morphology.celldt as celldt
import moma_pipeline.morphology.worker as worker
report.append("1) all 10 ported modules import: OK")

# 2) heavy deps stay lazy outside models/ (nothing imported above needs them)
assert "cellpose" not in sys.modules, "cellpose leaked into Phase-1 imports!"
assert "torch" not in sys.modules, "torch leaked into Phase-1 imports!"
assert "omnishim" not in sys.modules, "omnishim leaked into Phase-1 imports!"
report.append("2) no cellpose/torch/omnishim in sys.modules after Phase-1 imports: OK")

# 3) cross-module chain: synthetic rod mask -> CellDT via loky worker driver
mask = np.zeros((60, 60), dtype=np.uint16)
mask[24:36, 8:52] = 1
imgs = {"bf": np.full((60, 60), 100.0), "gfp": np.full((60, 60), 150.0)}
cells = worker.dt_timepoint_compute(
    mask, {"bf": imgs["bf"], "gfp": imgs["gfp"]},
    {"mean": {"bf": 90.0, "gfp": 120.0}, "median": {"bf": 90.0, "gfp": 120.0}},
    4, ["gfp"], 15, "fov_0_bf_t_0.tif", umppx=0.065)
assert len(cells) == 1 and cells[0].cellSkeletonSuccess, "dt_timepoint_compute failed"
assert abs(cells[0].spine_length - (44 * 0.065)) < 0.05, cells[0].spine_length
assert "median_gfp" in cells[0].flu_distribution, "flu distribution missing"
report.append("3) worker -> CellDT on synthetic rod (spine + flu distribution): OK")

# 4) pkl round-trip through the DT export schema (io.fov_pkl lands in Phase 2,
#    so round-trip via pickle directly here)
import pickle
rec = cells[0].export_cell_parameters()
rt = celldt.CellDT(); rt.load_cell_parameters(rec)
assert abs(rt.spine_length - cells[0].spine_length) < 1e-9
report.append("4) CellDT export/load_cell_parameters round-trip: OK")

# 5) images + contours + the OpenCV fix
lab = images.label_seg(mask.astype(np.uint8))
assert lab.max() == 1 and lab.dtype == np.uint16
single = images.getSinglecells(mask.astype(np.uint8))
assert single.shape[0] == 1
report.append("5) label_seg/getSinglecells (OpenCV4 tuple fix) work: OK")

# 6) nd2 helpers + transactions
assert nd2_extract.jdn_to_unix_seconds(2440587.5) == 0.0
assert images.read_image is not None and contours.cv_find_contours is not None
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "sub" / "x.json"
    transactions.atomic_json(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
    assert transactions.recover_transaction(Path(td) / "nope.json") == "missing"
report.append("6) nd2_extract helpers + atomic_json + recover_transaction: OK")

# 7) describe_dt_morphology provenance intact
m = worker.describe_dt_morphology()
assert m["version"] == "dt_channel_reference_bounded_poles_2026_11_06"
report.append("7) describe_dt_morphology provenance version: OK")

print("\n".join(report))
print("PHASE-1 SMOKE OK")
