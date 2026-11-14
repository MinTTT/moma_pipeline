# -*- coding: utf-8 -*-
"""ND2-to-Cell-Analysis Pipeline with DT-based Morphology (thin wrapper).

  1. ``nd2_extract.extract_nd2_to_fov_tiffs`` writes the ND2 file into the
     standard FOV folder layout ``save_dir/fov_<p>/<channel>/...tif`` with
     Unix-epoch-second ``acq_time`` metadata and a resumable
     ``nd2_extraction_manifest.json``.
  2. ``moma_pipeline.detection.pipeline.run_pipeline`` analyzes ``save_dir``
     exactly as it would any extracted FOV directory (same channel resolution,
     DT morphology, exports, logging and run-parameter provenance). The ND2
     extraction summary is added to the ``[input]`` run-parameter group.

All DT analysis options are shared with ``moma_pipeline.detection.pipeline``.
Historical flags are kept as aliases: ``-c`` for the cell detection channel,
``--skip_processed`` for ``--skip_existing`` and ``--skip_extracted`` for
``--extract skip``.

@author: CHU Pan
@date: 2026-06-03 (refactored 2026-11-01)
@e-mail: pan_chu@outlook.com
"""

from __future__ import annotations

import argparse
import os

from moma_pipeline.io.nd2_extract import (
    EXTRACT_MODES, extract_nd2_to_fov_tiffs, summarize_manifest,
)
from moma_pipeline.detection.pipeline import (
    add_dt_pipeline_args, dt_pipeline_kwargs_from_args, run_pipeline,
)


def default_save_dir(nd2_path):
    """``<nd2_dir>/Moma_process_<nd2_basename>``."""
    name = os.path.splitext(os.path.basename(nd2_path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(nd2_path)),
                        f'Moma_process_{name}')


def build_parser():
    parser = argparse.ArgumentParser(
        description='ND2-to-Cell-Analysis DT Pipeline for Mother Machine Data: '
                    'extract ND2 to per-FOV TIFFs, then run the FOV DT pipeline.')
    parser.add_argument('--nd2_path', '-n', type=str, default=None,
                        help='Path to the input ND2 file.')
    parser.add_argument('--save_dir', '-s', type=str, default=None,
                        help='Output directory for TIFFs and analysis results '
                             '(default: <nd2_dir>/Moma_process_<nd2_name>).')
    parser.add_argument('--extract', choices=EXTRACT_MODES, default='auto',
                        help='auto: reuse a complete matching extraction, '
                             'resume a partial one, otherwise extract; '
                             'force: re-extract everything; skip: reuse '
                             'existing FOV folders without extracting.')
    parser.add_argument('--skip_extracted', action='store_true',
                        help='Deprecated alias for --extract skip.')
    parser.add_argument('--extract_only', action='store_true',
                        help='Only extract the ND2 file; do not run analysis.')
    add_dt_pipeline_args(
        parser,
        channel_flags=('--cell_detection_channel', '-c', '-d'),
        skip_existing_flags=('--skip_existing', '--skip_processed'))
    return parser


def resolve_cli(args, parser):
    """Return (nd2_path, save_dir, extract_mode) from parsed arguments."""
    if args.nd2_path is None:
        parser.error('--nd2_path is required.')
    extract_mode = args.extract
    if args.skip_extracted:
        if extract_mode not in ('auto', 'skip'):
            parser.error('--skip_extracted conflicts with --extract '
                         f'{extract_mode}.')
        print('Note: --skip_extracted is deprecated; use --extract skip.')
        extract_mode = 'skip'
    save_dir = args.save_dir or default_save_dir(args.nd2_path)
    return args.nd2_path, save_dir, extract_mode


def run_nd2_pipeline(nd2_path, save_dir, extract_mode='auto',
                     extract_only=False, **dt_kwargs):
    """Extract ``nd2_path`` into ``save_dir`` and run the FOV DT pipeline."""
    print(f"ND2 file: {nd2_path}")
    print(f"Output directory: {save_dir}")
    print(f"Extraction mode: {extract_mode}")
    manifest = extract_nd2_to_fov_tiffs(nd2_path, save_dir, mode=extract_mode)
    if extract_only:
        print('Extraction finished (--extract_only); analysis not run.')
        return manifest

    extra_input = {
        'nd2_path': os.path.abspath(nd2_path),
        'nd2_extract_mode': extract_mode,
        'nd2_extraction': summarize_manifest(manifest),
    }
    run_pipeline(fovs_dir=save_dir, extra_input=extra_input, **dt_kwargs)
    return manifest


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    nd2_path, save_dir, extract_mode = resolve_cli(args, parser)
    try:
        run_nd2_pipeline(nd2_path, save_dir, extract_mode=extract_mode,
                         extract_only=args.extract_only,
                         **dt_pipeline_kwargs_from_args(args))
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
