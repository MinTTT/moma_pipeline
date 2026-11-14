"""python -m moma_pipeline - list the available commands."""
from __future__ import annotations

COMMANDS = (
    ("moma-nd2-pipeline", "extract an ND2 file to per-FOV TIFFs, then analyze"),
    ("moma-fov-pipeline", "analyze an existing FOV folder"),
    ("moma-cell-parameters", "recompute DT morphology from exported pkls"),
    ("moma-flu-distribution", "backfill fluorescence distributions from pkls"),
    ("moma-view-timelapse", "lazy napari timelapse browser over raw TIFF trees"),
    ("moma-view-fov", "napari viewer for exported pkls (+ live re-segmentation)"),
)


def main() -> int:
    print("moma_pipeline commands:\n")
    width = max(len(name) for name, _ in COMMANDS)
    for name, desc in COMMANDS:
        print(f"  {name:<{width}}  {desc}")
    print("\nRun any command with --help for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
