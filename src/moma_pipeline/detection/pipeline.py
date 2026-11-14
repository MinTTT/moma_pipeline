"""
DT-based FOV Processing Pipeline for Mother Machine Data.

Processes FOV directories (extracted TIFFs) through:
  1. Channel detection (Cellpose side-groove model)
  2. Cell detection (Cellpose cell model)
  3. DT-based morphology extraction (CellDT from moma_pipeline.morphology)
  4. CSV + pickle export

Compared to the legacy contour pipeline, this pipeline uses the distance-transform
based morphology algorithm. Cellpose detection stays on the main thread while
DT morphology/export runs in a simple background thread so GPU detection for
the next FOV can overlap CPU morphology for the previous FOV.

This module is the import-light orchestration layer: torch/cellpose/omnishim
and the parallel-written detection helpers (channels/fov/dt_utils) are imported
lazily inside the functions that use them, so argparse handling and FOV
discovery work without the model stack loaded.

@author: CHU Pan
@date: 2026-05-19
@e-mail: pan_chu@outlook.com
"""

from __future__ import annotations

import os
import sys
import time
import platform
import argparse
from threading import Thread

from moma_pipeline.core.fov_utils import get_time_now, parse_fov_index
from moma_pipeline.core.logging import SafeLogRecorder
from moma_pipeline.morphology.worker import describe_dt_morphology


MAX_NUM_CELLS_ANALYZED = 4

# =============================================================================
# Run-parameter provenance
# =============================================================================

def build_run_parameters(fovs_dir, cell_detection_channel, resolved_channel,
                         moma_cp_model, threads_num, max_cell_number,
                         cell_niter, skip_existing, dt_jobs_requested,
                         dt_jobs, cpu_count, keep_models_loaded,
                         flu_distribution, flu_distribution_channels,
                         flu_distribution_bins, extra_input=None):
    """Collect every analysis parameter into one JSON-serializable dict.

    ``moma_cp_model`` is used read-only: its eval parameter dicts are copied
    after any ``cell_niter`` override, and its model paths / resolved engine
    come from ``describe_models()``. ``extra_input`` (e.g. ND2 extraction
    provenance from the ND2 wrapper) is merged into the ``input`` group.
    """
    from moma_pipeline.models import cellpose_version, gpu_available
    auto_channel = (cell_detection_channel is None
                    or str(cell_detection_channel).strip().lower() == 'auto')

    return {
        'run': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'command': ' '.join(sys.argv),
            'script': os.path.abspath(__file__),
            'working_directory': os.getcwd(),
            'python': platform.python_version(),
            'platform': platform.platform(),
            'cellpose_version': cellpose_version(),
            'gpu': bool(gpu_available()),
        },
        'input': {
            'fovs_dir': os.path.abspath(fovs_dir),
            'cell_detection_channel': ('auto' if auto_channel
                                       else str(cell_detection_channel)),
            'cell_detection_channel_first_fov': resolved_channel,
            **(extra_input or {}),
        },
        'models': moma_cp_model.describe_models(),
        'eval_cell_params': dict(moma_cp_model.eval_cell_params),
        'eval_channel_params': dict(moma_cp_model.eval_channel_params),
        'pipeline': {
            'algorithm': 'DT (cell_morphology_compat.CellDT)',
            'morphology': describe_dt_morphology(),
            'threads': threads_num,
            'max_cells_per_channel': max_cell_number,
            'cell_niter': cell_niter,
            'skip_existing': bool(skip_existing),
            'dt_jobs_requested': dt_jobs_requested,
            'dt_jobs_resolved': dt_jobs,
            'cpu_count': cpu_count,
            'dt_backend': 'loky',
            'dt_background_threads': 1,
            'keep_models_loaded': bool(keep_models_loaded),
            'flu_distribution': bool(flu_distribution),
            'flu_distribution_channels': flu_distribution_channels,
            'flu_distribution_bins': flu_distribution_bins,
        },
    }


def format_run_parameters(run_parameters):
    """Render the run-parameters dict as a readable log block."""
    lines = ['=' * 20 + ' Run parameters ' + '=' * 20]
    for group, values in run_parameters.items():
        if isinstance(values, dict):
            lines.append(f'[{group}]')
            for key, value in values.items():
                lines.append(f'  {key}: {value}')
        else:
            lines.append(f'{group}: {values}')
    lines.append('=' * 56)
    return '\n'.join(lines)

# =============================================================================
# FOV discovery
# =============================================================================

def discover_fov_dirs(fovs_dir):
    """Return FOV subdirectory paths ('fov' in the name), sorted by index."""
    fov_name_list = [d for d in os.listdir(fovs_dir)
                     if os.path.isdir(os.path.join(fovs_dir, d)) and 'fov' in d]
    fovs_path = [os.path.join(fovs_dir, name) for name in fov_name_list]
    fovs_path.sort(key=parse_fov_index)
    return fovs_path


def get_completed_fov_names(export_dir):
    """FOV names with BOTH ``_cells_df.csv`` and ``_exported_data.pkl``.

    The CSV is written before the pickle, so a CSV alone may come from an
    export that failed half-way; such FOVs are reprocessed.
    """
    if not os.path.isdir(export_dir):
        return set()
    files = os.listdir(export_dir)
    csv_fovs = {f[:-len('_cells_df.csv')] for f in files
                if f.endswith('_cells_df.csv')}
    pkl_fovs = {f[:-len('_exported_data.pkl')] for f in files
                if f.endswith('_exported_data.pkl')}
    return csv_fovs & pkl_fovs


# =============================================================================
# Orchestration
# =============================================================================

def analyze_single_fov_dt(fov_dir, cell_detection_channel, moma_cp_model,
                           threads_num, max_cell_number, log_recorder,
                           dt_jobs, flu_distribution=False,
                           flu_distribution_channels=None,
                           flu_distribution_bins=15,
                           keep_models_loaded=False):
    """Run channel and cell detection for one FOV using MomaFOV.

    Returns MomaFOV after detection succeeds. DT morphology and export are
    scheduled separately so the main thread can start detecting the next FOV.
    """
    from moma_pipeline.detection.channels import (
        resolve_channel_name_in_fov as resolve_channel_name)
    from moma_pipeline.detection.fov import MomaFOV

    fov_name = os.path.basename(fov_dir)
    # Resolve channel name case-insensitively; None/'auto' auto-detects a
    # 'bf'/'phase' channel from the FOV's folder names
    resolved_channel = resolve_channel_name(fov_dir, cell_detection_channel)
    if resolved_channel is not None:
        auto = (cell_detection_channel is None
                or str(cell_detection_channel).strip().lower() == 'auto')
        log_recorder.write_log(
            f'{get_time_now()} Moma_CPM_DT -> {fov_name}: '
            f"Cell detection channel: '{resolved_channel}'"
            + (' (auto-detected)' if auto else ''))
    try:
        fov_obj = MomaFOV(
            fov_directory=fov_dir,
            cell_detection_opticalChannel_name=resolved_channel,
            MoMa_CP_model=moma_cp_model,
            log_recorder=log_recorder,
            max_cell_number=max_cell_number,
            dt_jobs=dt_jobs,
            flu_distribution=flu_distribution,
            flu_distribution_channels=flu_distribution_channels,
            flu_distribution_bins=flu_distribution_bins,
            keep_models_loaded=keep_models_loaded)
    except Exception as e:
        log_recorder.write_log(
            f'{get_time_now()} Moma_CPM_DT -> {fov_name}: Failed to initialize - {e}')
        return None

    try:
        channel_ret = fov_obj.detect_and_crop_channels()
        if channel_ret == 1:
            log_recorder.write_log(
                f'{get_time_now()} Moma_CPM_DT -> {fov_name}: '
                'No channels detected, skipping.')
            return None

        fov_obj.detect_cells(threads_num_max=threads_num)
        return fov_obj
    finally:
        if not keep_models_loaded:
            moma_cp_model.release_models('all')

def _calculate_and_export_worker(fov_obj, errors, log_recorder):
    """Run CPU-only DT morphology/export and record background failures."""
    from moma_pipeline.detection.dt_utils import calculate_and_export_dt
    try:
        calculate_and_export_dt(fov_obj)
    except Exception as exc:
        errors.append((fov_obj.fov_name, exc))
        log_recorder.write_log(
            f'{get_time_now()} Moma_CPM_DT -> {fov_obj.fov_name}: '
            f'DT calculation/export failed - {exc}')
        import traceback
        log_recorder.write_log(traceback.format_exc())


def run_pipeline(fovs_dir, cell_detection_channel, cell_model_path,
                 channel_model_path, threads_num, max_cell_number,
                 cell_niter=None,
                 skip_existing=False, dt_jobs=0, flu_distribution=False,
                 flu_distribution_channels=None, flu_distribution_bins=15,
                 keep_models_loaded=False, extra_input=None):
    """Main orchestration for DT-based FOV processing.

    ``extra_input`` is an optional JSON-serializable dict merged into the
    ``[input]`` run-parameter group (used by the ND2 wrapper for extraction
    provenance).
    """
    from moma_pipeline.detection.channels import (
        resolve_channel_name_in_fov as resolve_channel_name)
    from moma_pipeline.detection.dt_utils import resolve_dt_jobs

    if not os.path.exists(fovs_dir):
        raise FileNotFoundError(f"FOVs directory not found: {fovs_dir}")
    if not os.path.exists(cell_model_path):
        raise FileNotFoundError(f"Cell model not found: {cell_model_path}")
    if not os.path.exists(channel_model_path):
        raise FileNotFoundError(f"Channel model not found: {channel_model_path}")

    dt_jobs_requested = dt_jobs
    dt_jobs, cpu_count = resolve_dt_jobs(dt_jobs, reserved_cpus=threads_num)

    print(f"\n{'='*60}")
    print(f"DT-Based FOV Processing Pipeline")
    print(f"{'='*60}")
    print(f"FOVs directory: {fovs_dir}")

    # Discover FOVs (sorted by numeric FOV index)
    fovs_path = discover_fov_dirs(fovs_dir)
    if len(fovs_path) == 0:
        print(f"{get_time_now()} No FOV directories found in {fovs_dir}. Exiting.")
        return

    # check if the cell detection channel exists in the first FOV
    # (in auto mode this also fails fast when no known BF/phase folder exists)
    resolved_channel = resolve_channel_name(fovs_path[0], cell_detection_channel)
    auto_channel = (cell_detection_channel is None
                    or str(cell_detection_channel).strip().lower() == 'auto')
    if auto_channel:
        print(f"Auto-detected cell detection channel (first FOV): "
              f"{resolved_channel} (resolved per FOV)")
    else:
        print(f"Resolved cell detection channel: {resolved_channel}")
    print(f"Cell detection channel: "
          f"{cell_detection_channel if cell_detection_channel is not None else 'auto'}")
    print(f"Cell model: {cell_model_path}")
    print(f"Channel model: {channel_model_path}")
    print(f"Cellpose cell niter: {cell_niter}")
    print(f"Threads: {threads_num}")
    print(f"Max cells per channel: {max_cell_number}")
    print(f"CPU cores: {cpu_count}")
    print("DT background thread: 1")
    print(f"DT jobs: {dt_jobs} (loky process pool)")
    print(f"Keep Cellpose models loaded: {keep_models_loaded}")
    print(f"Fluorescence distribution: {flu_distribution}")
    if flu_distribution:
        if flu_distribution_channels:
            print(f"Fluorescence distribution channels: {flu_distribution_channels}")
        else:
            print("Fluorescence distribution channels: all non-cell-detection channels")
        print(f"Fluorescence distribution bins: {flu_distribution_bins}")
    print()

    # Create the log first (SafeLogRecorder handles PermissionError on
    # Windows) so model loading and every later stage is recorded
    export_dir = os.path.join(fovs_dir, 'exported_data')
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(export_dir, f"{timestamp}_Moma_CPM_DT.log")
    log_recorder = SafeLogRecorder(log_file)
    log_recorder.write_log(
        f"DT Pipeline started. FOVs dir: {os.path.abspath(fovs_dir)}")

    # Filter already-processed FOVs
    print(f"{get_time_now()} Step 1: Discovering FOV directories...")
    if skip_existing:
        existing_fovs = get_completed_fov_names(export_dir)
        fovs_path = [p for p in fovs_path
                     if os.path.basename(p) not in existing_fovs]
        print(f"  Found {len(fovs_path)} FOV(s) to process "
              f"({len(existing_fovs)} already done, skipped).")
        log_recorder.write_log(
            f"skip_existing: {len(existing_fovs)} FOV(s) with both "
            f"_cells_df.csv and _exported_data.pkl skipped.")
    else:
        print(f"  Found {len(fovs_path)} FOV(s) to process.")

    # Initialize models (the log recorder is already active, so model-load
    # and engine-resolution messages are recorded in the log file)
    print(f"\n{get_time_now()} Step 2: Loading Cellpose models...")
    from moma_pipeline.models import create_engine
    moma_cp_model = create_engine(cell_model_path=cell_model_path,
                                  channel_model_path=channel_model_path,
                                  log_recorder=log_recorder)
    print(f"Channel eval params: {dict(moma_cp_model.eval_channel_params)} "
          f"(source: {getattr(moma_cp_model, 'channel_params_source', '-')})")
    if cell_niter is not None:
        # more iterations help Cellpose follow the flow field along long cells
        moma_cp_model.eval_cell_params['niter'] = cell_niter

    # Record every analysis parameter before processing starts
    run_parameters = build_run_parameters(
        fovs_dir=fovs_dir,
        cell_detection_channel=cell_detection_channel,
        resolved_channel=resolved_channel,
        moma_cp_model=moma_cp_model,
        threads_num=threads_num,
        max_cell_number=max_cell_number,
        cell_niter=cell_niter,
        skip_existing=skip_existing,
        dt_jobs_requested=dt_jobs_requested,
        dt_jobs=dt_jobs,
        cpu_count=cpu_count,
        keep_models_loaded=keep_models_loaded,
        flu_distribution=flu_distribution,
        flu_distribution_channels=flu_distribution_channels,
        flu_distribution_bins=flu_distribution_bins,
        extra_input=extra_input,
    )
    log_recorder.write_log(format_run_parameters(run_parameters))

    if keep_models_loaded:
        moma_cp_model.load_models('all')
    else:
        moma_cp_model.load_models('channel')
    print("  Models loaded.")

    # Process FOVs
    print(f"\n{get_time_now()} Step 3: Processing {len(fovs_path)} FOV(s)...")
    print(f"{'-'*60}")

    success_count = 0
    failed_count = 0
    skipped_count = 0
    calc_thread = None
    calc_task_count = 0
    calc_errors = []

    for fov_i, fov_dir in enumerate(fovs_path):
        fov_name = os.path.basename(fov_dir)
        log_recorder.write_log(
            f"[{fov_i + 1}/{len(fovs_path)}] Processing {fov_name}...")
        print(f"[{fov_i + 1}/{len(fovs_path)}] {fov_name}...", end=" ")

        try:
            result = analyze_single_fov_dt(
                fov_dir, cell_detection_channel, moma_cp_model,
                threads_num, max_cell_number, log_recorder, dt_jobs,
                flu_distribution=flu_distribution,
                flu_distribution_channels=flu_distribution_channels,
                flu_distribution_bins=flu_distribution_bins,
                keep_models_loaded=keep_models_loaded)
            if result is None:
                skipped_count += 1
                print("SKIPPED (no channels)")
            else:
                if calc_thread is not None and calc_thread.is_alive():
                    print("waiting previous DT calc...", end=" ")
                    calc_thread.join()

                calc_thread = Thread(
                    target=_calculate_and_export_worker,
                    args=(result, calc_errors, log_recorder),
                    name=f'DT-calc-{fov_name}')
                calc_thread.start()
                calc_task_count += 1
                success_count += 1
                print("OK (DT calc thread started)")
        except Exception as e:
            failed_count += 1
            print(f"FAILED: {e}")
            log_recorder.write_log(f"  FAILED: {e}")
            import traceback
            log_recorder.write_log(traceback.format_exc())

    if calc_thread is not None:
        print(f"\n{'-'*60}")
        print(f"Waiting for {calc_task_count} DT calculation thread(s) to finish...")
        calc_thread.join()

    if calc_errors:
        failed_count += len(calc_errors)
        success_count -= len(calc_errors)
        print("\nDT calculation/export failures:")
        for fov_name, exc in calc_errors:
            print(f"  {fov_name}: {exc}")

    # Summary
    print(f"{'-'*60}")
    print(f"\n{get_time_now()} Pipeline completed.")
    print(f"  Total FOVs: {len(fovs_path)}")
    print(f"  Successful: {success_count}")
    print(f"  Skipped:    {skipped_count}")
    print(f"  Failed:     {failed_count}")

    log_recorder.write_log(
        f"Pipeline completed. Total: {len(fovs_path)}, "
        f"Success: {success_count}, Skipped: {skipped_count}, "
        f"Failed: {failed_count}")

    moma_cp_model.release_models('all')
    print(f"\n{get_time_now()} Models released. Log saved to: {log_file}")


# =============================================================================
# Shared CLI arguments
# =============================================================================

def add_dt_pipeline_args(parser, channel_flags=('--cell_detection_channel', '-d'),
                         skip_existing_flags=('--skip_existing',)):
    """Add the DT analysis options shared by the FOV and ND2 entry points.

    ``channel_flags`` / ``skip_existing_flags`` let wrappers keep historical
    aliases; the argparse destinations stay ``cell_detection_channel`` and
    ``skip_existing``. The default model paths come from
    ``moma_pipeline.models.defaults`` (imported lazily so this module stays
    importable without the model stack).
    """
    from moma_pipeline.models.defaults import (
        DEFAULT_CELL_MODEL_PATH, DEFAULT_CHANNEL_MODEL_PATH)
    parser.add_argument(*channel_flags, dest='cell_detection_channel',
                        type=str, default=None,
                        help="Optical channel name for cell detection. "
                             "Default (omit or pass 'auto'): auto-detect from "
                             "each FOV's channel folder names, case-insensitive "
                             "(see moma_pipeline.detection.channels."
                             "CELL_DETECTION_CHANNEL_PATTERNS).")
    parser.add_argument('--threads', type=int, default=1,
                        help='Max threads for cell detection (default: 1).')
    parser.add_argument('--max_cells', type=int, default=MAX_NUM_CELLS_ANALYZED,
                        help=f'Max cells per channel per time point '
                             f'(default: {MAX_NUM_CELLS_ANALYZED}).')
    parser.add_argument('--cell_model_path', type=str, default=DEFAULT_CELL_MODEL_PATH,
                        help=f'Path to Cellpose cell model (default: {DEFAULT_CELL_MODEL_PATH}).')
    parser.add_argument('--channel_model_path', type=str, default=DEFAULT_CHANNEL_MODEL_PATH,
                        help=f'Path to the channel-detection model '
                             f'(default: {DEFAULT_CHANNEL_MODEL_PATH}). Evaluation '
                             'parameters follow the model family unless given '
                             'explicitly (diameter 30 / min_size 15 for the new '
                             'Fine-tuned Deep model, diameter 60 / min_size 1000 '
                             'for legacy channel models). Pass '
                             '<root>/cellpose_sideGrooves_20250708-2 to roll back.')
    parser.add_argument('--cell_niter', type=int, default=None,
                        help='Iterations (niter) for Cellpose cell mask recomputation. '
                             'Default is None (Cellpose auto: 200 at diameter 30, scales as '
                             '200*diameter/30). Increase (e.g. 400) to better segment '
                             'long cells.')
    parser.add_argument(*skip_existing_flags, dest='skip_existing',
                        action='store_true',
                        help='Skip FOVs that already have both _cells_df.csv '
                             'and _exported_data.pkl outputs.')
    parser.add_argument('--dt_jobs', type=int, default=-1,
                        help='Worker processes for DT morphology (loky pool). '
                             '0 = auto use CPU cores not reserved by --threads; '
                             '-1 = all CPUs (default).')
    parser.add_argument(
        '--keep_models_loaded', action='store_true',
        help='Keep both Cellpose models resident across FOVs. This avoids '
             'per-FOV model reloads but uses more GPU memory. Without this '
             'flag, each model is released after its detection phase.')
    parser.add_argument('--flu_distribution', action='store_true',
                        help='Compute fluorescence distributions along the DT cell axis. '
                             'Without --flu_distribution_channels, all non-cell-detection '
                             'optical channels are analyzed.')
    parser.add_argument('--flu_distribution_channels', nargs='+', default=None,
                        help='Explicit optical channel names for fluorescence distribution, '
                             'for example: --flu_distribution_channels yellow. '
                             'Comma-separated values are also accepted.')
    parser.add_argument('--flu_distribution_bins', type=int, default=15,
                        help='Relative-position bins for fluorescence distribution '
                             '(default: 15; use 31 only for legacy reproduction).')
    return parser


def dt_pipeline_kwargs_from_args(args):
    """Translate parsed shared DT options into ``run_pipeline`` kwargs."""
    channel = args.cell_detection_channel
    if channel is not None and str(channel).strip().lower() == 'auto':
        channel = None
    flu_distribution_channels = normalize_channel_args(args.flu_distribution_channels)
    flu_distribution = args.flu_distribution or flu_distribution_channels is not None
    return dict(
        cell_detection_channel=channel,
        cell_model_path=args.cell_model_path,
        channel_model_path=args.channel_model_path,
        threads_num=args.threads,
        max_cell_number=args.max_cells,
        cell_niter=args.cell_niter,
        skip_existing=args.skip_existing,
        dt_jobs=args.dt_jobs,
        flu_distribution=flu_distribution,
        flu_distribution_channels=flu_distribution_channels,
        flu_distribution_bins=args.flu_distribution_bins,
        keep_models_loaded=args.keep_models_loaded,
    )


def normalize_channel_args(channel_args):
    """Normalize repeated and comma-separated channel CLI arguments."""
    if channel_args is None:
        return None
    channels = []
    for item in channel_args:
        for channel in str(item).split(','):
            channel = channel.strip()
            if channel:
                channels.append(channel)
    return channels or None
