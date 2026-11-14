"""Small shared helpers (ported from utilities.py / Moma_CPM_FOV.py)."""
from __future__ import annotations

import os
import time
from typing import Optional


def get_time_now(format: Optional[str] = None) -> str:
    """Current time as 'YYYY-MM-DD HH:MM:SS' (format '1'/None) or 'YYYYMMDD_HHMMSS' (format '2')."""
    if format is None:
        format = '1'
    if format == '1':
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if format == '2':
        return time.strftime("%Y%m%d_%H%M%S", time.localtime())
    raise ValueError(f"Unknown time format: {format!r}")


def parse_fov_index(fov_path):
    """Parse the FOV index from the FOV path ('fov_3' -> 3, 'fov_2-1' -> 2.1)."""
    fov_name = os.path.basename(fov_path)
    label = fov_name.split('_')[-1].strip('fov')
    if '-' in label:
        label = label.split('-')
        return float(label[0] + '.' + label[1])
    return int(label)
