"""Log recorders (ported from Moma_CPM_FOV.py / moma_dt_pipeline_utils.py)."""
from __future__ import annotations

from typing import Optional

from .fov_utils import get_time_now


class LogRecorder:
    """Print a message and append it to a log file (when a path is given)."""

    def __init__(self, log_file_path: Optional[str] = None):
        if log_file_path is not None:
            self.log_file_path = log_file_path
            with open(self.log_file_path, 'w') as f:
                f.write(f'{get_time_now()} Log file created.\n')
        else:
            self.log_file_path = None

    def write_log(self, log_message: str):
        print(log_message)
        if self.log_file_path:
            with open(self.log_file_path, 'a') as f:
                f.write(f'{get_time_now()} {log_message}\n')
        return None


class SafeLogRecorder:
    """Wrapper around LogRecorder that falls back to stdout-only logging."""

    def __init__(self, log_file_path=None):
        try:
            self._inner = LogRecorder(log_file_path)
        except PermissionError:
            print("Warning: Cannot create log file, using stdout only.")
            self._inner = LogRecorder(None)

    def write_log(self, message):
        try:
            self._inner.write_log(message)
        except PermissionError:
            self._inner = LogRecorder(None)
            print(message)
