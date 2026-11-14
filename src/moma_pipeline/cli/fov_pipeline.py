"""DT-based FOV processing pipeline CLI.

Ported from the ``__main__`` surface of
Moma_cellPose/Moma_CPM_FOV_DT_pipeline.py into
``moma_pipeline.cli.fov_pipeline`` (the shared orchestration lives in
``moma_pipeline.detection.pipeline``).
"""
from __future__ import annotations

import argparse
import os

from moma_pipeline.detection.pipeline import (
    add_dt_pipeline_args, dt_pipeline_kwargs_from_args, normalize_channel_args,
    run_pipeline,
)

__all__ = ['build_parser', 'main', 'normalize_channel_args']


def build_parser():
    parser = argparse.ArgumentParser(
        description='DT-based FOV Processing Pipeline for Mother Machine Data.')

    parser.add_argument('--fovs_dir', '-f', type=str, default=None,
                        help='Directory containing FOV subdirectories to process.')
    parser.add_argument('--test', action='store_true',
                        help='Run on FOV_example_data directory.')
    add_dt_pipeline_args(parser)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    fovs_dir = None

    if args.test:
        # Would resolve to <package-root>/../../FOV_example_data; example
        # data is not bundled with the package.
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        example_dir = os.path.join(package_root, '..', '..', 'FOV_example_data')
        print(f"Error: --test example data is not bundled with moma_pipeline "
              f"(expected at {example_dir}). "
              f"Use --fovs_dir to pass a FOV directory.")
        return 2

    if args.fovs_dir is not None:
        fovs_dir = args.fovs_dir

    if fovs_dir is None:
        parser.error("Either --fovs_dir or --test must be specified.")

    try:
        run_pipeline(fovs_dir=fovs_dir, **dt_pipeline_kwargs_from_args(args))
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
