# Moma_pipeline — Porting & Architecture Plan

**Target**: a standalone, src-layout Python package `moma_pipeline` that runs the
complete MotherMachine analysis pipeline — from data (ND2 file or FOV TIFF folder)
to final morphology export — re-using `omnishim` (G:\python_code\omnishim) for all
deep-learning model usage.

**Non-goals**: model training / model development (stays in Moma_cellPose / omnishim).

---

## 1. What the review found (port surface)

### 1.1 CLI / APP entry points to port (the 6 user-facing scripts)

| # | Source script | Kind | Role | Port difficulty |
|---|---------------|------|------|-----------------|
| 1 | Moma_CPM_FOV_DT_pipeline.py (24 KB) | CLI | Main FOV pipeline orchestration (discovery, run-parameter provenance, background DT thread, skip-existing) | Low — thin orchestrator over the detection core |
| 2 | Moma_CPM_ND2_DT_pipeline.py (5 KB) | CLI | Thin wrapper: ND2 -> FOV TIFFs (nd2_extract) -> FOV pipeline | Low |
| 3 | Moma_cell_parameters_from_pkls.py (49 KB) | CLI | Headless recompute of DT morphology from *_exported_data.pkl with journaled transactions; no models | Medium — 867 ln, many conventions to preserve |
| 4 | tests/flu_distribution_from_pkls/Moma_flu_distribution_from_pkls.py (23 KB) | CLI | Headless backfill of fluorescence-distribution profiles from pkls | Low-Medium — sys.path hack + file-path provenance hashing |
| 5 | MomaViewer/lazy_napari_timelapse_viewer.py (36 KB) | APP | Lazy napari timelapse browser over raw TIFF trees (dask-free, LRU cache) | Low — fully self-contained (napari/qtpy/tifffile/numpy) |
| 6 | Moma_CPM_FOV_viewer.py (16 KB) | APP | napari viewer for exported pkls with **live cellpose re-segmentation** | High — ~15 mutable module globals, hardcoded ./CPModels + machine-specific defaults, raw cellpose usage |

### 1.2 Library modules the 6 scripts depend on (~390 KB of code)

| Source module | Size | Role | Model-free? |
|---------------|------|------|-------------|
| Moma_CPM_FOV.py | 72 KB / 1429 ln | MomaFOV (channel detection/cropping, drift correction, time-series cell detection), MoMa_CPM (model manager), LogRecorder, legacy exports | **No** — cellpose/torch/omnishim |
| moma_dt_pipeline_utils.py | 19 KB | MomaFOV_DT (loky DT morphology + DT pkl schema), SafeLogRecorder, channel auto-detection, dt_jobs resolution | Indirect |
| moma_dt_worker.py | 12 KB | describe_dt_morphology() provenance + dt_timepoint_compute() loky worker | Yes (deliberately torch/cellpose-free — keep it that way) |
| moma_dt_exports.py | 5 KB | CSV + flu-distribution collectors | Yes (pandas) |
| moma_dt_kernels.py | 3 KB | numba kernels for DT mesh/cap geometry | Yes |
| nd2_extract.py | 19 KB | ND2 -> save_dir/fov_<p>/<channel>/*.tif + resumable manifest | Yes (lazy `nd2` reader) |
| cell_morphology_compat.py | 37 KB | CellDT — DT morphology cell class (the 'Cell' replacement) | Yes (numpy/numba) |
| cell_morphology_voronoi.py | 130 KB | DT medial-axis morphology core | Yes (numba) |
| utilities.py | 36 KB | read_image, driftcorr, chamber boxes, cropbox, get_time_now | Yes (cv2) |
| cell_coord_utility.py | 50 KB | Legacy contour Cell + cv helpers + cubic solvers | **DROPPED (decision D2)** — not ported; DT-only pipeline |
| cellUtility.py | 10 KB | cv_find_contours | Yes |
| moma_file_transactions.py | 27 KB | Stdlib-only journaled multi-file publish with backups/recovery | Yes |

### 1.3 Key facts that shape the design

- **omnishim already ships the model wrapper we need**: `omnishim.compat.moma.MoMaCellposeEngine`
  is a documented, tested drop-in for the *model-management half* of `MoMa_CPM`
  (same constructor/attributes/methods/log lines, engine auto-dispatch
  `auto|cellpose|omnishim`, `niter->omni_niter` remapping, eval locks, catalog-based
  default-model resolution via `OMNISHIM_MODEL_DIRS` / `omnishim models add-dir`).
- Therefore `MoMa_CPM` is **not ported** — it is replaced by a thin pipeline-side
  facade over `MoMaCellposeEngine`. The detection code (`MomaFOV`/`MomaFOV_DT`) only
  ever touches the duck-typed engine interface: `load_models` / `release_models` /
  `eval_images` / `describe_models` / `eval_cell_params` / `eval_channel_params`.
- **Exactly 3 touch-points use DL models** in the whole port surface:
  1. `MoMa_CPM` (replaced by omnishim compat engine) — pipeline,
  2. `Moma_CPM_FOV_viewer.load_model` — raw `cellpose.models.CellposeModel(gpu=True)`,
  3. `cell_coord_utility.cell_segmentation` — raw `CellposeModel(model_type=...)` (legacy helper).
  All three get routed through `moma_pipeline.models` so **no other module imports
  cellpose/torch/omnishim directly** (enforced by a source-scan test).
- **Environment**: conda env `Moma_cellpose`, Python 3.11.13, cellpose 4.2.1.1,
  torch 2.8.0.dev+cu128 (CUDA OK), omnishim 0.5.0 installed editable.
- `Moma_cellPose` has **no packaging at all** (no pyproject/setup.py); the new
  project starts clean.

---

## 2. Target package structure (src layout)

```
G:\python_code\Moma_pipeline\
├── pyproject.toml                  # name = "moma-pipeline", src layout, console scripts
├── README.md
├── environment.yml                 # frozen from Moma_cellPose env + omnishim + viewer extras
├── docs\
│   └── PORTING_PLAN.md             # this file
├── src\moma_pipeline\
│   ├── __init__.py                 # __version__, lazy public API
│   ├── __main__.py                 # python -m moma_pipeline → prints CLI help
│   │
│   ├── models\                     # ★ the only place that may import cellpose/torch/omnishim
│   │   ├── __init__.py             #   lazy exports: create_engine, SingleModelSegmenter
│   │   ├── defaults.py             #   DEFAULT_CELL_MODEL / DEFAULT_CHANNEL_MODEL (catalog names),
│   │   │                           #   DEFAULT_CELL_PARAMS / DEFAULT_CHANNEL_PARAMS
│   │   ├── engine.py               #   MomaModelEngine(MoMaCellposeEngine) + create_engine()
│   │   └── segmenter.py            #   SingleModelSegmenter — one-model service for the FOV viewer
│   │
│   ├── detection\                  # was Moma_CPM_FOV.py + moma_dt_pipeline_utils.py
│   │   ├── __init__.py
│   │   ├── fov.py                  #   single DT-native MomaFOV (merged MomaFOV+MomaFOV_DT, D2)
│   │   ├── channels.py             #   auto-detect cell-detection channel, resolve_channel_name_in_fov
│   │   ├── dt_utils.py             #   resolve_dt_jobs, calculate_and_export_dt
│   │   └── pipeline.py             #   run_pipeline() orchestration shared by FOV/ND2 CLIs
│   │
│   ├── morphology\                 # model-free
│   │   ├── __init__.py
│   │   ├── celldt.py               #   was cell_morphology_compat.py (CellDT)
│   │   ├── voronoi_dt.py           #   was cell_morphology_voronoi.py
│   │   │                           #   (legacy_cell.py DROPPED per decision D2 — DT-only pipeline)
│   │   ├── kernels.py              #   was moma_dt_kernels.py
│   │   └── worker.py               #   was moma_dt_worker.py (keep torch/cellpose-free!)
│   │
│   ├── io\                         # model-free
│   │   ├── __init__.py
│   │   ├── images.py               #   was utilities.py (read_image, driftcorr, chamber boxes;
│   │   │                           #   fix skimage.external.tifffile, drop bioformats, param n_jobs)
│   │   ├── contours.py             #   was cellUtility.py
│   │   ├── nd2_extract.py          #   as-is (lazy nd2 import kept)
│   │   ├── dt_exports.py           #   was moma_dt_exports.py
│   │   ├── fov_pkl.py              #   DT + legacy pkl schemas (export_Moma_fov_data_dt, load_*, legacy)
│   │   └── transactions.py         #   was moma_file_transactions.py (stdlib-only, as-is)
│   │
│   ├── cli\                        # 4 headless CLIs
│   │   ├── __init__.py
│   │   ├── fov_pipeline.py         #   was Moma_CPM_FOV_DT_pipeline.py
│   │   ├── nd2_pipeline.py         #   was Moma_CPM_ND2_DT_pipeline.py
│   │   ├── cell_parameters.py      #   was Moma_cell_parameters_from_pkls.py
│   │   └── flu_distribution.py     #   was tests/.../Moma_flu_distribution_from_pkls.py
│   │
│   ├── viewers\                    # 2 napari apps (import lazily; napari only in [viewer] extra)
│   │   ├── __init__.py
│   │   ├── lazy_timelapse.py       #   was MomaViewer/lazy_napari_timelapse_viewer.py
│   │   └── fov_viewer.py           #   was Moma_CPM_FOV_viewer.py (class-based refactor, model via models/)
│   │
│   └── core\
│       ├── __init__.py
│       ├── logging.py              #   LogRecorder, SafeLogRecorder
│       └── fov_utils.py            #   parse_fov_index, get_time_now, fov discovery helpers
│
└── tests\
    ├── unit\                       # ported pure-module tests
    ├── integration\               # pipeline parity tests
    └── fixtures\                  # small FOV_example_data subset (or symlink policy)
```

### 2.1 Old -> new name map (for reviewers)

| Old | New |
|-----|-----|
| Moma_CPM_FOV_DT_pipeline.py | moma_pipeline/cli/fov_pipeline.py |
| Moma_CPM_ND2_DT_pipeline.py | moma_pipeline/cli/nd2_pipeline.py |
| Moma_cell_parameters_from_pkls.py | moma_pipeline/cli/cell_parameters.py |
| tests/flu_distribution_from_pkls/Moma_flu_distribution_from_pkls.py | moma_pipeline/cli/flu_distribution.py |
| MomaViewer/lazy_napari_timelapse_viewer.py | moma_pipeline/viewers/lazy_timelapse.py |
| Moma_CPM_FOV_viewer.py | moma_pipeline/viewers/fov_viewer.py |
| Moma_CPM_FOV.py | split -> detection/fov.py, core/logging.py, core/fov_utils.py (**MoMa_CPM deleted**, replaced by models/engine.py) |
| moma_dt_pipeline_utils.py | split -> detection/fov_dt.py, detection/channels.py, io/fov_pkl.py |
| moma_dt_worker.py | morphology/worker.py |
| moma_dt_exports.py | io/dt_exports.py |
| moma_dt_kernels.py | morphology/kernels.py |
| nd2_extract.py | io/nd2_extract.py |
| cell_morphology_compat.py | morphology/celldt.py |
| cell_morphology_voronoi.py | morphology/voronoi_dt.py |
| cell_coord_utility.py | **DROPPED** (decision D2 — DT-only; `MomaFOV` references `CellDT` directly) |
| cellUtility.py | io/contours.py |
| utilities.py | io/images.py |
| moma_file_transactions.py | io/transactions.py |

---

## 3. Model wrapper design (`moma_pipeline.models`)

Goal: pipeline code never imports `cellpose`/`torch`/`omnishim` directly; all model
usage goes through this facade, which delegates to omnishim (which itself wraps
cellpose). Three use-cases exist:

1. **Pipeline two-model manager** (channel model + cell model, lazy load/release,
   eval-param dicts, GPU sharing, provenance for run parameters).
2. **Viewer single-model segmenter** (pick a checkpoint by catalog name or path,
   run one-off evals with user params).
3. **Legacy helper** `cell_segmentation` in morphology/legacy_cell.py (same as #2,
   kept for backward compatibility).

### 3.1 `models/engine.py` — pipeline manager

```python
from omnishim.compat.moma import (MoMaCellposeEngine, DEFAULT_CELL_PARAMS,
                                  DEFAULT_CHANNEL_PARAMS, cell_eval_kwargs)

from .defaults import DEFAULT_CELL_MODEL, DEFAULT_CHANNEL_MODEL

class MomaModelEngine(MoMaCellposeEngine):
    """Pipeline-facing model manager.

    Thin specialization of omnishim's MoMa-compatible engine:
    - package defaults are *catalog names* (resolved by omnishim.catalog),
      never CWD-relative ./CPModels paths;
    - no import-time GPU probing (gpu=None -> CUDA when available, resolved lazily);
    - identical duck-typed surface to the old MoMa_CPM, so detection/fov.py
      needs almost no changes.
    """
    def __init__(self, cell_model_path=None, channel_model_path=None, *,
                 cell_model_params=None, channel_model_params=None,
                 cell_model_engine="auto", log_recorder=None, gpu=None):
        super().__init__(
            cell_model_path=cell_model_path or DEFAULT_CELL_MODEL,
            channel_model_path=channel_model_path or DEFAULT_CHANNEL_MODEL,
            cell_model_params=(dict(DEFAULT_CELL_PARAMS) if cell_model_params is None
                               else cell_model_params),
            channel_model_params=(dict(DEFAULT_CHANNEL_PARAMS) if channel_model_params is None
                                  else channel_model_params),
            cell_model_engine=cell_model_engine,
            log_recorder=log_recorder, gpu=gpu)

def create_engine(**kwargs) -> MomaModelEngine:
    return MomaModelEngine(**kwargs)
```

### 3.2 `models/defaults.py`

```python
# Model paths are assigned DIRECTLY (decision D3): this package takes explicit
# paths and does not depend on the omnishim catalog being configured.
# Override the root with the MOMA_MODEL_ROOT env var, or pass paths explicitly.
import os

DEFAULT_MODEL_ROOT = os.environ.get("MOMA_MODEL_ROOT",
                                    r"G:\python_code\Moma_cellPose\CPModels")
DEFAULT_CELL_MODEL_PATH = os.path.join(DEFAULT_MODEL_ROOT, "models",
                                       "cellpose_cells_20260525")
DEFAULT_CHANNEL_MODEL_PATH = os.path.join(DEFAULT_MODEL_ROOT,
                                          "cellpose_sideGrooves_20250708-2")
```

Eval-param defaults are re-exported from `omnishim.compat.moma`
(`diameter=30, cellprob_threshold=0, flow_threshold=0.4, niter=None` for cells;
`diameter=60, ..., min_size=1000` for channels) so there is ONE source of truth.
Non-existent model paths may still resolve through omnishim's catalog as a
fallback (that is omnishim's concern, decision D3).

### 3.3 `models/segmenter.py` — single-model service (viewer)

```python
class SingleModelSegmenter:
    """Load one checkpoint (catalog name or path) and run eval on demand.

    Built on omnishim.pipeline.Segmenter.from_checkpoint + omnishim.compat.resolve_engine,
    so an omnishim checkpoint automatically uses its native encoder/decoder.
    """
    def __init__(self, model_ref, *, engine="auto", gpu=None, log_recorder=None): ...
    @classmethod
    def from_name_or_path(cls, ref, **kwargs): ...
    def eval(self, image, **eval_params): ...   # returns masks (first output)
    def release(self): ...
```

Replaces `Moma_CPM_FOV_viewer.load_model` -> raw `cellpose.models.CellposeModel(gpu=True)`
and `cell_coord_utility.cell_segmentation`.

### 3.4 Enforcement

- A unit test scans `src/moma_pipeline/**.py` and fails on `import cellpose`,
  `import torch`, `from cellpose`, `from omnishim` outside `models/`.
- `models/__init__.py` uses lazy attribute resolution (same pattern as
  `omnishim.compat`), so importing `moma_pipeline` never pulls torch/cellpose.

---

## 4. CLI / APP entry points

Console scripts (pyproject `[project.scripts]`):

| Command | Function | Notes |
|---------|----------|-------|
| `moma-fov-pipeline` | cli.fov_pipeline:main | Same flags as today (`--fovs_dir/-f`, `--test`, plus all DT options). `--cell_model_path`/`--channel_model_path` now accept **catalog names or paths**; defaults are catalog names. |
| `moma-nd2-pipeline` | cli.nd2_pipeline:main | `--nd2_path/-n`, `--save_dir/-s`, `--extract auto|force|skip`, `--extract_only`; deprecated aliases kept. |
| `moma-cell-parameters` | cli.cell_parameters:main | Full argparse surface preserved verbatim (incl. `BEGIN_BATCH_JSON` stdout marker, exit-code semantics). |
| `moma-flu-distribution` | cli.flu_distribution:main | `--pkl_dir`, `--output_dir`, `--channels`, `--n-bins`, `--jobs`, `--skip_existing/--overwrite`. |
| `moma-view-timelapse` | viewers.lazy_timelapse:main | napari lazy timelapse viewer. |
| `moma-view-fov` | viewers.fov_viewer:main | + new optional positional `exported_data_dir` argument replacing the hardcoded default; model dropdown lists omnishim catalog entries. |

Run-parameter provenance (`build_run_parameters`) is kept; `models` section comes
from `engine.describe_models()`; `script`/`command` fields become the new
console-script paths (old script path kept as `legacy_script` for one release).

---

## 5. Dependencies & environment

```toml
[project]
name = "moma-pipeline"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "numpy", "scipy", "pandas", "scikit-image", "opencv-python-headless",
  "tifffile", "tqdm", "joblib>=1.4", "numba", "nd2", "natsort", "fastremap",
  "cellpose>=4.2,<5", "torch", "torchvision", "omnishim",
]
[project.optional-dependencies]
viewer = ["napari[pyqt6]", "magicgui", "qtpy", "pyqtgraph"]
dev = ["pytest", "pytest-timeout"]
```

- `omnishim` is consumed as an installed package: `pip install -e G:\python_code\omnishim`
  (already done in this env) then `pip install -e G:\python_code\Moma_pipeline --no-deps`
  during development — same pattern omnishim's own docs describe.
- `environment.yml` documents the full conda env for reproducibility (mirrors
  Moma_cellPose's, minus training-only pins, plus omnishim).
- Model discovery at runtime: `omnishim models add-dir G:\python_code\Moma_cellPose\CPModels`
  (once per machine) or `$env:OMNISHIM_MODEL_DIRS`. No `./CPModels` CWD-relative
  paths remain anywhere.

---

## 6. Porting phases & acceptance criteria

### Phase 0 — scaffold (half day)
- pyproject.toml, src tree, README, .gitignore, environment.yml, `pip install -e` into env `Moma_cellpose`.
- Acceptance: `import moma_pipeline` works; `moma-fov-pipeline --help` shows (stub).

### Phase 1 — model-free foundation (1–2 days)
- Port: io/images.py, io/contours.py, io/transactions.py, io/nd2_extract.py,
  morphology/voronoi_dt.py, morphology/celldt.py, morphology/kernels.py,
  morphology/worker.py, io/dt_exports.py, core/logging.py, core/fov_utils.py.
- Fixes applied: `skimage.external.tifffile` -> `tifffile`; drop bioformats branch from
  xpreader (quarantine to `io/_legacy_xpreader.py` or delete); parameterize
  driftcorr `n_jobs=32`; inline the one lazy `read_image` use in nd2_extract.
- Port unit tests: test_nd2_extract, test_cell_morphology_compat, kernels/tests,
  transaction tests.
- Acceptance: `pytest tests/unit` green on the ported subset.

### Phase 2 — models facade + detection core (2–3 days)
- models/{defaults,engine,segmenter}.py; detection/{fov,fov_dt,channels}.py,
  io/fov_pkl.py, cli/{fov_pipeline,nd2_pipeline}.py.
- **Single DT-native `MomaFOV`** (decision D2): `MomaFOV` and `MomaFOV_DT` are
  merged into one class that references `CellDT` directly; the legacy
  contour `calculate_cell_parameters`, legacy exports (`export_Moma_fov_data`,
  `export_cells_to_dataframe`) and the `cell_coord_utility` import are deleted.
- Engine default creation through `models.create_engine()`; removal of the
  `GPU_USED` import-time global (replaced by lazy `models.gpu_available()`);
  the `sifht_correction` typo is kept as a deprecated alias.
- Acceptance (parity): run old vs new pipeline on a small FOV subset
  (FOV_example_data/LDJ_Kstarv_recover) with the same checkpoints ->
  `*_cells_df.csv` equal within tolerance, pkl `info_dict` equal,
  run-parameter JSONs diff-clean except run/script fields.
  Also port test_model_residency.py + test_dt_run_parameters.py.

> **Phase 2 completed 2026-11-07 — parity verified.** Full pipeline run on
> `FOV_example_data/SeqA_SpotDet/fov_0` (11 timepoints x 3 channels, real models):
> old `Moma_CPM_FOV_DT_pipeline.py` vs new `moma-fov-pipeline` produced **identical**
> output — 371 cells x 38 CSV columns equal to 1e-9, all segmentation masks and
> channel images byte-identical, `time_points_seconds`, background fluorescence and
> `channel_info.txt` identical, and all 371 cell-parameter records equal.
> Evidence: `scripts_dev/parity_check.py`, logs in `scripts_dev/_sandbox/*.log`.

### Phase 3 — recompute & flu CLIs (1–2 days)
- cli/cell_parameters.py + cli/flu_distribution.py.
- Preserve byte-level conventions: journal naming, backup dir, CSV identity-column
  string coercion, pickle HIGHEST_PROTOCOL, BEGIN_BATCH_JSON marker, exit codes.
- Provenance hashing switches from `PROJECT_ROOT/<file>.py` paths to
  `importlib.util.find_spec(...).origin`.
- Port tests: tests/cell_parameters_from_pkls/*, tests/flu_distribution_from_pkls/*.
- Acceptance: golden-folder test — recompute a reference pkl, byte-compare
  validated outputs against reference run.

> **Phases 3 & 4 completed 2026-11-07.** cell_parameters: default run refuses
> publish with 12 invalid cells and an error report identical to the old script;
> `--allow_invalid` publish produces CSV/pkl byte-identical to the old script.
> flu_distribution: pkl identical, summary identical modulo run-specific fields.
> Viewers: lazy_timelapse ported verbatim (+ `[root]` startup arg);
> fov_viewer refactored to class-based FovViewer (globals removed, latent
> UnboundLocalError fixed, models via SingleModelSegmenter, hardcoded Y:\ path
> and ./CPModels defaults replaced by package defaults + --model_dir).

### Phase 4 — viewers (1–2 days)
- viewers/lazy_timelapse.py — nearly mechanical port (self-contained).
- viewers/fov_viewer.py — refactor into a class-based widget; model dropdown fed
  from `omnishim.catalog.list_models()`; live re-segmentation via
  `SingleModelSegmenter`; `--exported_data_dir` CLI arg; remove hardcoded
  Y:\ default and global-state handlers.
- Acceptance: both apps launch from console scripts; viewer reproduces old
  behavior on a sample exported_data pkl.

### Phase 5 — hardening & cleanup (ongoing)
- Source-scan test (no cellpose/torch/omnishim imports outside models/).
- Delete legacy interactive `__main__` blocks (hardcoded Y:/Linux paths, input()).
- `cell_coord_utility` is gone (decision D2): nothing to reroute.
- CI: pytest on Windows, README usage docs, version bump policy.
- Optional later: absorb `Moma_CPM_ND2_pipeline_parallel.py` (parallel ND2 variant).

---

## 7. Compatibility guarantees

1. **Output contract unchanged**: same filenames (`<fov>_cells_df.csv`,
   `<fov>_exported_data.pkl`, `<fov>_flu_distribution.pkl`, channel masks/TIFs,
   channel_info.txt), same CSV columns, same pkl dict schema (dict-based — no
   pickled class instances), same manifest schema.
2. **CLI flags preserved** (including deprecated aliases) — old commands keep
   working by adding the console script name in front.
3. **Old DT exports remain readable**: `load_Moma_fov_data_dt` is ported, so
   DT pkls produced by the old repo open in the new package (schema unchanged).
   Pre-DT legacy pkls are out of scope (decision D2 — the legacy loader
   `load_Moma_fov_data` is not ported).
4. **Engine behavior identical**: omnishim's compat engine reproduces MoMa_CPM's
   dispatch rules, eval-param mapping and log-line formats by design (it is
   tested against them in omnishim's own test suite).

---

## 8. Risks & open decisions

| Risk / decision | Mitigation |
|---|---|
| `omnishim` and `moma_pipeline` both editable — version skew | pin `omnishim>=0.5` in deps; run omnishim's compat tests in CI |
| numba `cache=True` writes __pycache__ inside src | fine under editable installs; document |
| loky worker must stay torch/cellpose-free | enforced by the source-scan test on morphology/worker.py imports |
| OpenCV findContours convention (cv2 major version) | pin `opencv-python-headless>=4.8`; keep mixed-convention code as-is initially, fix in cleanup |
| Old pkls pickle classes from old module paths | exports are dict-based; verified — no class instances persisted |
| FOV viewer behavior change (globals refactor) | keep old script untouched in Moma_cellPose during transition; compare side-by-side |
| Model catalog not configured on a new machine | clear error message from omnishim (`ModelNotFoundError` with instructions) + README quickstart |
| ~~Decision~~ LOCKED 2026-11-07: package name | `moma_pipeline` / `moma-pipeline` (D1) |
| ~~Decision~~ LOCKED 2026-11-07: legacy contour morphology | DROPPED — DT-only `MomaFOV` referencing `CellDT` (D2) |
| ~~Decision~~ LOCKED 2026-11-07: model discovery | direct model paths in this package; catalog bootstrap is omnishim's concern (D3) |
| Decision: Python floor | `>=3.10` (env is 3.11; omnishim also requires 3.10+) |
