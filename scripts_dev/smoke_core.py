"""Phase-0 smoke test: core helpers and entry-point stubs."""
import moma_pipeline
import moma_pipeline.core.logging as L
import moma_pipeline.core.fov_utils as U

print("version", moma_pipeline.__version__, "from", moma_pipeline.__file__)
assert L.LogRecorder is not None and L.SafeLogRecorder is not None
print("time1:", U.get_time_now(), "| time2:", U.get_time_now("2"))
assert U.parse_fov_index("x/fov_12") == 12
assert U.parse_fov_index("fov_2-1") == 2.1

# stubs exit(2) with not-yet-ported message
import subprocess, sys
from pathlib import Path
scripts = Path(sys.prefix) / "Scripts"
for cmd in ("moma-fov-pipeline", "moma-nd2-pipeline", "moma-cell-parameters",
            "moma-flu-distribution", "moma-view-timelapse", "moma-view-fov"):
    exe = scripts / (cmd + ".exe")
    assert exe.exists(), f"missing console script: {exe}"
    p = subprocess.run([str(exe), "--help"], capture_output=True, text=True)
    assert p.returncode == 0 and "not yet ported" in p.stdout, (cmd, p.returncode, p.stdout)
print("all 6 entry-point stubs respond to --help: OK")
print("SMOKE OK")
