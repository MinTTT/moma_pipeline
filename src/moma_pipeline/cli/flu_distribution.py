"""Backfill production DT fluorescence profiles from read-only local exports.

Run from any directory with the project's interpreter, for example::

    python -m moma_pipeline.cli.flu_distribution --pkl_dir exported_data

The distribution pickle is the production bare side-channel/time/cell mapping.
Diagnostics and provenance live in separate JSON files. No segmentation or DT
reconstruction is performed. See METHODS_FLU_DISTRIBUTION_FROM_PKLS.md.
"""
from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util as importlib_util
import json
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile

import numpy as np

from moma_pipeline.morphology.celldt import CellDT

METHOD_VERSION = "0.1.2"
SOURCE_SUFFIX = "_exported_data.pkl"


def _module_source(modname):
    """Resolve a module's source file for provenance hashing."""
    spec = importlib_util.find_spec(modname)
    return Path(spec.origin) if spec and spec.origin else None


def _positive_integer(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def normalize_channels(channels):
    """Normalize comma/space-separated CLI or API names, preserving request order."""
    if channels is None:
        return None
    if isinstance(channels, str):
        channels = [channels]
    result = []
    seen = set()
    for entry in channels:
        for name in re.split(r"[,\s]+", str(entry).strip()):
            if name and name.lower() not in seen:
                result.append(name)
                seen.add(name.lower())
    return result or None


def restore_cell(params):
    """Restore a successful saved CellDT without running morphology.

    Mesh-rib midpoints are the authoritative centerline, including coincident
    centers in round geometry. Missing legacy geometry_mode uses CellDT's loader
    inference. Invalid/missing saved geometry raises ValueError.
    """
    if not isinstance(params, Mapping):
        raise ValueError("cell parameters must be a mapping")
    if not params.get("cellSkeletonSuccess", False):
        raise ValueError("upstream_invalid_geometry")
    if params.get("mesh") is None:
        raise ValueError("missing_mesh")
    mesh = np.array(params["mesh"], dtype=np.float64, copy=True)
    if (mesh.ndim != 2 or mesh.shape[0] < 2 or mesh.shape[1] != 4
            or not np.all(np.isfinite(mesh))):
        raise ValueError("invalid_mesh")
    if params.get("endpoint_radii") is None:
        raise ValueError("missing_endpoint_radii")
    radii = np.asarray(params["endpoint_radii"], dtype=np.float64)
    if radii.shape != (2,) or np.any(~np.isfinite(radii)) or np.any(radii < 0):
        raise ValueError("invalid_endpoint_radii")
    if params.get("umppx") is None:
        raise ValueError("missing_umppx")
    umppx = float(params["umppx"])
    if not np.isfinite(umppx) or umppx <= 0:
        raise ValueError("invalid_umppx")
    cell = CellDT()
    cell.load_cell_parameters(dict(params))
    cell.mesh = mesh
    cell.centerline = (mesh[:, :2] + mesh[:, 2:4]) / 2.0
    cell.spine = cell.centerline
    cell.skeleton = cell.centerline
    cell._endpoint_radii = tuple(float(r) for r in radii)
    cell.umppx = umppx
    cell.flu_distribution = {}
    if cell.geometry_mode not in {"rod", "short_capsule", "round", "branched_primary", "channel_axis"}:
        raise ValueError(f"invalid_geometry_mode: {cell.geometry_mode!r}")
    if cell.geometry_mode != "round":
        steps = np.linalg.norm(np.diff(cell.centerline, axis=0), axis=1)
        if np.any(~np.isfinite(steps)) or np.any(steps <= 1e-12):
            raise ValueError("invalid_centerline_tangent")
    # This also validates the selected round rib and the production profile length.
    cell._fluorescence_profile_line()
    return cell


def _report():
    return {
        "status": "ok",
        "counts": {"side_channels": 0, "time_points": 0, "cells_seen": 0,
                   "cells_computed": 0, "cells_invalid": 0, "cells_failed": 0,
                   "cells_without_channels": 0, "channel_profiles": 0},
        "warnings": [], "errors": [],
    }


def _issue(report, kind, code, message, **context):
    report[kind].append({"code": code, "message": str(message), **context})


def _finish(report):
    report["warning_count"] = len(report["warnings"])
    report["error_count"] = len(report["errors"])
    report["status"] = "error" if report["errors"] else (
        "empty" if not report["counts"]["cells_computed"] else "ok")
    return report


def _cell_label(params):
    value = params.get("cell_index")
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError("missing or invalid cell_index")
    label = int(value)
    if label < 1 or (not isinstance(value, str) and label != value):
        raise ValueError(f"invalid cell_index: {value!r}")
    return label


def _compute_timepoint(side, time, records, payload, detection, channels, n_bins):
    report = _report()
    report["counts"]["time_points"] = 1
    context = {"side_channel": str(side), "time_point": str(time)}
    result = {}
    if not isinstance(records, (list, tuple)):
        _issue(report, "errors", "invalid_cell_records", "expected a list of saved cells", **context)
        return side, time, result, report
    report["counts"]["cells_seen"] = len(records)
    if not records:
        return side, time, result, report
    masks = payload.get("fov_cells_mask_dict", {}).get(side, {}).get(time)
    images = payload.get("fov_channels_image_dict", {}).get(side, {}).get(time)
    if masks is None or not isinstance(images, Mapping):
        report["counts"]["cells_failed"] = len(records)
        _issue(report, "errors", "missing_timepoint_inputs", "missing label mask or optical images", **context)
        return side, time, result, report
    masks = np.asarray(masks)
    if masks.ndim != 2:
        report["counts"]["cells_failed"] = len(records)
        _issue(report, "errors", "invalid_mask", "label mask must be two-dimensional", **context)
        return side, time, result, report
    available = list(images)
    lower_to_channel = {str(name).lower(): name for name in available}
    selected = []
    if channels is None:
        selected = [name for name in available if str(name).lower() != str(detection).lower()]
    else:
        for name in channels:
            matched = lower_to_channel.get(name.lower())
            if matched is None:
                _issue(report, "warnings", "missing_channel", f"requested optical channel {name!r} is absent", **context)
            else:
                selected.append(matched)
    background_by_time = payload.get("images_background_fluorescence", {})
    background = background_by_time.get(time, {}) if isinstance(background_by_time, Mapping) else {}
    if not isinstance(background, Mapping):
        _issue(report, "errors", "invalid_background", "per-time background must be a mapping", **context)
        report["counts"]["cells_failed"] = len(records)
        return side, time, result, report
    for name in selected:
        saved = background.get(name)
        if saved is None or (isinstance(saved, dict) and saved.get("mean") is None):
            _issue(report, "warnings", "zero_background", f"missing background mean for {name!r}; using production zero fallback", **context)
    seen = set()
    for params in records:
        cell_context = dict(context)
        try:
            if not isinstance(params, Mapping):
                raise ValueError("cell parameters must be a mapping")
            label = _cell_label(params)
            cell_context["cell_index"] = label
            if label in seen:
                raise ValueError(f"duplicate cell_index: {label}")
            seen.add(label)
            if not params.get("cellSkeletonSuccess", False) or params.get("geometry_mode") == "invalid":
                report["counts"]["cells_invalid"] += 1
                _issue(report, "warnings", "upstream_invalid_geometry", "saved cell geometry was invalid; omitted", **cell_context)
                continue
            cell = restore_cell(params)
            cell_mask = masks == label
            if not np.any(cell_mask):
                raise ValueError(f"cell_index {label} is absent from label mask")
            # Do not allow image indexing to hide a mismatched crop shape.
            for name in selected:
                image = images[name]
                if np.ndim(image) != 2 or tuple(np.shape(image)) != tuple(masks.shape):
                    raise ValueError(f"cell mask and fluorescence image shapes do not match: {name!r}")
            if not selected:
                report["counts"]["cells_without_channels"] += 1
                continue
            cell.channel_imgs = images
            cell.background_fluorescence = background
            distribution = cell.compute_flu_distribution(selected, n_bins=n_bins, cell_mask=cell_mask)
            if not distribution:
                raise ValueError("production calculation returned an empty distribution")
            result[label] = distribution
            report["counts"]["cells_computed"] += 1
            report["counts"]["channel_profiles"] += len(selected)
        except Exception as exc:
            report["counts"]["cells_failed"] += 1
            _issue(report, "errors", "cell_failed", f"{type(exc).__name__}: {exc}", **cell_context)
    return side, time, result, report


def _ordered_thread_results(function, tasks, jobs):
    """At most jobs tasks in flight, yielded in original traversal order."""
    if jobs == 1:
        for task in tasks:
            yield function(*task)
        return
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = deque()
        tasks = iter(tasks)
        for _ in range(jobs):
            task = next(tasks, None)
            if task is None:
                break
            pending.append(pool.submit(function, *task))
        while pending:
            yield pending.popleft().result()
            task = next(tasks, None)
            if task is not None:
                pending.append(pool.submit(function, *task))


def compute_distributions(data, channels=None, n_bins=15, jobs=1):
    """Return (production bare mapping, diagnostics), preserving saved record order.

    Failed upstream cells are exclusions. Missing inputs, malformed saved geometry,
    duplicate labels and calculation failures are errors. The API returns valid
    partial profiles for inspection, but analyze_folder never saves such a partial
    distribution as a successful FOV output.
    """
    n_bins = _positive_integer(n_bins, "n_bins")
    jobs = _positive_integer(jobs, "jobs")
    channels = normalize_channels(channels)
    report = _report()
    report["parameters"] = {"flu_distribution_channels": channels,
                            "flu_distribution_bins": n_bins, "jobs": jobs}
    output = {}
    try:
        if not isinstance(data, Mapping):
            raise ValueError("export must be a mapping")
        info = data["info_dict"]
        payload = data["cells_sideChannel_image_dict"]
        if not isinstance(info, Mapping) or not isinstance(payload, Mapping):
            raise ValueError("info_dict and cells_sideChannel_image_dict must be mappings")
        if str(info.get("algorithm", "DT")).upper() != "DT":
            raise ValueError("only saved DT exports are supported")
        records = payload["fov_cells_pars_dict"]
        if not isinstance(records, Mapping):
            raise ValueError("fov_cells_pars_dict must be a mapping")
        detection = info.get("cell_detection_opticalChannel_name")
        if channels is None and detection is None:
            raise ValueError("missing cell_detection_opticalChannel_name for default channel selection")
        for field in ("fov_cells_mask_dict", "fov_channels_image_dict"):
            if not isinstance(payload.get(field), Mapping):
                raise ValueError(f"missing or invalid {field}")
        report["counts"]["side_channels"] = len(records)

        def tasks():
            for side, times in records.items():
                if not isinstance(times, Mapping):
                    raise ValueError(f"invalid cell time mapping for side channel {side!r}")
                for time, cells in times.items():
                    yield side, time, cells, payload, detection, channels, n_bins

        for side, time, distribution, local in _ordered_thread_results(_compute_timepoint, tasks(), jobs):
            if distribution:
                output.setdefault(side, {})[time] = distribution
            for name, count in local["counts"].items():
                report["counts"][name] += count
            report["warnings"].extend(local["warnings"])
            report["errors"].extend(local["errors"])
    except Exception as exc:
        _issue(report, "errors", "invalid_export", f"{type(exc).__name__}: {exc}")
    if not report["counts"]["cells_computed"]:
        _issue(report, "warnings", "zero_profiles", "no fluorescence profiles were computed")
    return output, _finish(report)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance():
    packages = {}
    for name in ("numpy", "numba", "scipy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    source_files = [Path(__file__)]
    for modname in ("moma_pipeline.morphology.celldt",
                    "moma_pipeline.morphology.voronoi_dt"):
        source_path = _module_source(modname)
        if source_path is not None:
            source_files.append(source_path)
    return {"method_version": METHOD_VERSION, "python": sys.version,
            "executable": sys.executable, "packages": packages,
            "command": list(sys.argv),
            "code_sha256": {str(path.resolve()): _sha256(path) for path in source_files}}


def _atomic_write(path, writer):
    """Write a same-directory temporary file then replace its destination atomically."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path, data):
    _atomic_write(path, lambda stream: stream.write(
        (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")))


def analyze_folder(pkl_dir, output_dir=None, channels=None, n_bins=15, jobs=1,
                   skip_existing=False, overwrite=False):
    """Process one source FOV at a time; save production PKLs and audit sidecars.

    Existing per-FOV output or sidecar requires explicit skip/overwrite. On a
    calculation error only a diagnostic sidecar is written; no partial PKL is
    written. A pre-existing PKL is retained on a failed overwrite and flagged.
    """
    n_bins = _positive_integer(n_bins, "n_bins")
    jobs = _positive_integer(jobs, "jobs")
    channels = normalize_channels(channels)
    if skip_existing and overwrite:
        raise ValueError("skip_existing and overwrite are mutually exclusive")
    source_dir = Path(pkl_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir else source_dir
    batch = {"status": "ok", "started_at": datetime.now(timezone.utc).isoformat(),
             "pkl_dir": str(source_dir), "output_dir": str(destination),
             "parameters": {"flu_distribution_channels": channels,
                            "flu_distribution_bins": n_bins, "jobs": jobs,
                            "skip_existing": bool(skip_existing), "overwrite": bool(overwrite)},
             "provenance": _provenance(), "files": [], "warnings": [], "errors": []}
    counts = {"files_seen": 0, "files_written": 0, "files_empty": 0,
              "files_skipped": 0, "files_failed": 0, "cells_seen": 0,
              "cells_computed": 0, "cells_invalid": 0, "cells_failed": 0,
              "cells_without_channels": 0, "channel_profiles": 0}
    batch["counts"] = counts
    if not source_dir.is_dir():
        batch["errors"].append({"code": "missing_directory", "message": f"input directory does not exist: {source_dir}"})
        batch["status"] = "error"
        return batch
    sources = sorted(source_dir.glob(f"*{SOURCE_SUFFIX}"), key=lambda path: path.name)
    destination.mkdir(parents=True, exist_ok=True)
    if not sources:
        batch["errors"].append({"code": "no_inputs", "message": f"no *{SOURCE_SUFFIX} files found"})
    for source in sources:
        counts["files_seen"] += 1
        fov = source.name[:-len(SOURCE_SUFFIX)]
        output = destination / f"{fov}_flu_distribution.pkl"
        sidecar = destination / f"{fov}_flu_distribution_summary.json"
        summary = {"fov": fov, "source_path": str(source), "output_path": str(output),
                   "summary_path": str(sidecar), "parameters": dict(batch["parameters"]),
                   "provenance": batch["provenance"], "distribution_written": False}
        batch["files"].append(summary)
        if output.exists() or sidecar.exists():
            if skip_existing:
                summary["status"] = "skipped"
                summary["reason"] = "existing output or summary"
                counts["files_skipped"] += 1
                continue
            if not overwrite:
                summary["status"] = "error"
                summary["errors"] = [{"code": "output_exists", "message": "output or summary exists; select --skip_existing or --overwrite"}]
                counts["files_failed"] += 1
                continue
        try:
            stat = source.stat()
            summary["source_sha256"] = _sha256(source)
            summary["source_size_bytes"] = stat.st_size
            summary["source_mtime_ns"] = stat.st_mtime_ns
            with open(source, "rb") as stream:
                data = pickle.load(stream)
            distribution, report = compute_distributions(data, channels, n_bins, jobs)
            del data
            summary.update(report)
            for key in ("cells_seen", "cells_computed", "cells_invalid", "cells_failed", "cells_without_channels", "channel_profiles"):
                counts[key] += report["counts"][key]
            # Verify input stability before publishing its derived distribution.
            if _sha256(source) != summary["source_sha256"]:
                _issue(summary, "errors", "source_changed", "source changed during processing")
                _finish(summary)
            if summary["errors"]:
                summary["existing_output_retained"] = output.exists()
                counts["files_failed"] += 1
            else:
                _atomic_write(output, lambda stream: pickle.dump(distribution, stream))
                summary["distribution_written"] = True
                summary["output_sha256"] = _sha256(output)
                counts["files_written"] += 1
                if summary["status"] == "empty":
                    counts["files_empty"] += 1
            del distribution
            _write_json(sidecar, summary)
        except Exception as exc:
            if summary.get("status") != "error":
                counts["files_failed"] += 1
            summary["status"] = "error"
            summary.setdefault("errors", []).append({"code": "file_failed", "message": f"{type(exc).__name__}: {exc}"})
            summary["existing_output_retained"] = output.exists()
            try:
                _write_json(sidecar, summary)
            except Exception as write_exc:
                summary["errors"].append({"code": "summary_write_failed", "message": str(write_exc)})
    batch["warning_count"] = sum(len(item.get("warnings", [])) for item in batch["files"]) + len(batch["warnings"])
    batch["error_count"] = sum(len(item.get("errors", [])) for item in batch["files"]) + len(batch["errors"])
    batch["status"] = "error" if batch["error_count"] else ("empty" if counts["files_empty"] == counts["files_seen"] else "ok")
    batch["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(destination / "flu_distribution_run_summary.json", batch)
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl_dir", "--pkl-dir", required=True, help="directory containing DT *_exported_data.pkl")
    parser.add_argument("--output_dir", "--output-dir", help="output directory (default: input directory)")
    parser.add_argument("--flu_distribution_channels", "--flu-distribution-channels", "--channels", nargs="+", help="case-insensitive names separated by spaces or commas; default: all non-detection channels")
    parser.add_argument("--flu_distribution_bins", "--flu-distribution-bins", "--n-bins", type=int, default=15)
    parser.add_argument("--jobs", type=int, default=1, help="positive number of threads within one loaded FOV")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--skip_existing", "--skip-existing", action="store_true")
    policy.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        batch = analyze_folder(args.pkl_dir, args.output_dir, args.flu_distribution_channels,
                               args.flu_distribution_bins, args.jobs, args.skip_existing, args.overwrite)
    except Exception as exc:
        print(f"Fluorescence distribution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": batch["status"], "counts": batch["counts"],
                      "warning_count": batch.get("warning_count", 0),
                      "error_count": batch.get("error_count", len(batch["errors"])),
                      "output_dir": batch["output_dir"]}, indent=2))
    return 1 if batch["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
