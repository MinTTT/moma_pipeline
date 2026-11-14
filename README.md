# moma-pipeline

Standalone MotherMachine data analysis pipeline, ported from the Moma_cellPose
project (analysis pipeline only, no model training).

**Pipeline**: ND2 file or FOV TIFF folder -> channel detection (Cellpose side-groove
model) -> cell detection (Cellpose cell model) -> distance-transform (DT) morphology
-> `*_cells_df.csv` / `*_exported_data.pkl` (plus optional fluorescence distributions).

All deep-learning model usage goes through omnishim
(`omnishim.compat.moma.MoMaCellposeEngine`); `moma_pipeline.models` is the only part
of the package that imports cellpose / torch / omnishim.

## Install (development)

```powershell
# in the conda env (e.g. Moma_cellpose), omnishim editable first:
pip install -e G:\python_code\omnishim
pip install -e G:\python_code\Moma_pipeline --no-deps
```

## Commands

| Command | Purpose |
|---|---|
| `moma-nd2-pipeline -n <file.nd2>` | extract ND2 to per-FOV TIFFs, then analyze |
| `moma-fov-pipeline -f <fovs_dir>` | analyze an existing FOV folder |
| `moma-cell-parameters --pkl_dir <dir>` | recompute DT morphology from exported pkls (journaled) |
| `moma-flu-distribution --pkl_dir <dir>` | backfill fluorescence distributions from pkls |
| `moma-view-timelapse` | lazy napari timelapse browser over raw TIFF trees |
| `moma-view-fov [exported_data_dir]` | napari viewer for exported pkls with live re-segmentation |

### Launch the FOV viewer

```powershell
moma-view-fov "G:\path\to\exported_data"
# Equivalent Python module entry point (not 'python moma-view-fov'):
python -m moma_pipeline.viewers.fov_viewer "G:\path\to\exported_data"
```

The time range is inclusive: `(0, 0)` shows one frame, and `(0, N-1)`
shows all N frames. Side-channel changes retain the selected range.
All optical channels (e.g. phase, green and red) are shown as separate,
synchronized napari layers. Missing channel images are displayed as zeros;
use each layer's controls to change visibility, contrast, opacity or colormap.
Segmentation and mask/image saving still use the configured detection channel.
If that channel is missing at the current time point, Run Model is skipped.

Viewer loading regressions are covered by
[the viewer tests](<tests/test_fov_viewer.py>):

```powershell
python -m pytest tests/test_fov_viewer.py -q
```

## Model paths

Model paths are assigned directly (see `moma_pipeline/models/defaults.py`).
The default root is `G:\python_code\Moma_cellPose\CPModels`; override with the
`MOMA_MODEL_ROOT` environment variable or the `--cell_model_path` /
`--channel_model_path` CLI flags. (omnishim model catalog remains available as a
fallback for name resolution; configuring it is omnishim concern.)

The default channel model is the Fine-tuned Deep channel model
`models/dino_omnipose_channel_deep_r001_20261108_s20260904` (SHA-256
`ec4ec9e582cef7cc54fa1b479771973f916936708a09f6e62435a6cce2d36b89`, model card
beside it as `.model_card.json`). omnishim recognises its `refine.*` tensors,
loads it with the DeepRefiner encoder and applies its model-matched evaluation
parameters (diameter 30, cellprob_threshold 0, flow_threshold 0.4, min_size 15).

**Rollback** to the previous channel model restores the old behaviour exactly:

```powershell
moma-fov-pipeline -f <fovs_dir> --channel_model_path <CPModels>\cellpose_sideGrooves_20250708-2
```

That checkpoint has no refinement head, so it is loaded by the plain cellpose
loader and evaluated with the historical parameters (diameter 60, cellprob 0,
flow 0.4, min_size 1000).
This applies to the CLI manager when no parameters are given. Direct MomaFOV
own-engine construction historically passes an empty dict; legacy models retain
that empty dict (stock Cellpose effective defaults) unchanged, while the deep
family receives the matched parameters above. The run log prints the effective
`Channel eval params ... (source: ...)` at start-up, and the run-parameters JSON
records them under `eval_channel_params` plus
`models.channel_model_params_source`.

## Verification status (2026-11-07)

All four phases are ported and verified against the old Moma_cellPose scripts on
real data (`FOV_example_data/SeqA_SpotDet/fov_0`, 11 timepoints x 3 channels):

| Check | Result |
|---|---|
| `moma-fov-pipeline` end-to-end run | 25 channel boxes, 10 channels with cells, 110 DT tasks, exports written |
| Pipeline parity vs old `Moma_CPM_FOV_DT_pipeline.py` | **bit-for-bit**: 371 cells x 38 CSV columns equal, all masks/images/records identical |
| `moma-cell-parameters` (default) | refuses publish, exit 1, 12 invalid cells — identical error report to old script |
| `moma-cell-parameters --allow_invalid` | published CSV/pkl **identical to old script**; same 13-column recompute delta vs pipeline CSV in both |
| `moma-flu-distribution` | pkl output identical to old script; summary identical modulo run-specific fields |
| Import-leak scan | zero cellpose/torch/omnishim imports outside `moma_pipeline/models/` |
| All 6 console scripts | respond to `--help` with the full original flag surface |

Evidence scripts: `scripts_dev/smoke_*.py`, `scripts_dev/parity_check.py`,
`scripts_dev/compare_cp_*.py`; run logs and demo outputs in `scripts_dev/_sandbox/`
(`SeqA_run` = one real FOV + exported data; safe to delete).

## Layout

See [docs/PORTING_PLAN.md](docs/PORTING_PLAN.md) for the architecture and the
old -> new module map.
