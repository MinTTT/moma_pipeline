"""DT job resolution and calculate/export orchestration helpers.

Ported verbatim from Moma_cellPose/moma_dt_pipeline_utils.py into
moma_pipeline.detection.dt_utils.
"""

import os


def resolve_dt_jobs(dt_jobs=0, reserved_cpus=0):
    """Resolve joblib thread count for one background DT morphology task.

    The pipeline intentionally keeps only one background DT task active because
    the DT task itself uses joblib threading. dt_jobs == 0 reserves the image
    detection thread budget and gives the remaining CPUs to DT morphology.
    Negative values are passed through to joblib, where -1 means all CPUs.
    """
    cpu_count = os.cpu_count() or 1

    dt_jobs = int(dt_jobs)
    if dt_jobs < 0:
        dt_jobs = cpu_count
    elif dt_jobs == 0:
        dt_jobs = max(1, cpu_count - max(0, int(reserved_cpus)))

    return dt_jobs, cpu_count


def calculate_and_export_dt(fov_obj):
    """Run DT morphology calculation and export for a detected FOV."""
    fov_obj.calculate_cell_parameters()
    fov_obj.export_cells_data()
