"""Recalculate DT cell measurements from saved segmentation exports.

Run from any directory. The default destination is <pkl_dir>/recalculated_data;
--in_place replaces matching PKL/CSV files with recoverable backups.
--umppx sets micrometers per pixel (default 0.065). No model is loaded.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from collections.abc import Mapping
from datetime import datetime, timezone
import importlib.util
from importlib import metadata
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
import uuid

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tifffile

from moma_pipeline.morphology.worker import describe_dt_morphology, dt_timepoint_compute
from moma_pipeline.io.dt_exports import export_cells_to_dataframe_dt, collect_flu_distribution_dt
from moma_pipeline.io.transactions import (
    atomic_json, publish_files, recover_transaction, sha256, transaction_lock,
)

METHOD_VERSION = "7.0.0"
# Compatibility alias for consumers of the former candidate version constant.
CONTOUR_POLE_METHOD_VERSION = METHOD_VERSION
SOURCE_SUFFIX = "_exported_data.pkl"
IDENTITY = ["channels", "time_point", "cell_index"]


class RecalculationError(ValueError):
    """Validation failure with structured diagnostics for the batch summary."""
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _scale(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("umppx must be finite and positive")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("umppx must be finite and positive")
    return value


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"missing or invalid {name}: expected a mapping")
    return value


def _label(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("invalid cell_index")
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid cell_index: {value!r}") from exc
    if result < 1 or (not isinstance(value, str) and result != value):
        raise ValueError(f"invalid cell_index: {value!r}")
    return result


def _side_token(value):
    # CSV parsers may turn integral side-channel IDs into floats.
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value)


def _key(side, time, label):
    return _side_token(side), str(time), _label(label)


def _diagnostic_component(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip(".")
    return text if text not in {"", ".", ".."} else "unnamed"


def _atomic_tiff(path, array):
    """Validate one TIFF before atomically replacing its destination."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.",
                suffix=".tif", delete=False) as stream:
            temporary = Path(stream.name)
            tifffile.imwrite(stream, array, photometric="minisblack")
            stream.flush()
            os.fsync(stream.fileno())
        restored = tifffile.imread(temporary)
        if restored.dtype != array.dtype or restored.shape != array.shape:
            raise ValueError(f"diagnostic TIFF dtype/shape changed: {path}")
        np.testing.assert_array_equal(restored, array)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_failed_cell_images(data, invalid_cells, output_dir, fov_name, *,
                              overwrite=False, source_sha256=None, umppx=0.065):
    """Export exact single-cell masks and the saved segmentation-channel image.

    The mask TIFF is uint8 0/1 for pixels equal to the reported cell ID, not the
    full multi-label mask. It is diagnostic data, not a replacement for the PKL.
    Each returned entry is JSON serializable and identifies its own metadata.
    """
    umppx = _scale(umppx)
    _mapping(data, "export")
    info = _mapping(data.get("info_dict"), "info_dict")
    payload = _mapping(data.get("cells_sideChannel_image_dict"),
                       "cells_sideChannel_image_dict")
    masks = _mapping(payload.get("fov_cells_mask_dict"), "fov_cells_mask_dict")
    images = _mapping(payload.get("fov_channels_image_dict"),
                      "fov_channels_image_dict")
    if not isinstance(invalid_cells, (list, tuple)):
        raise ValueError("invalid_cells must be a list of diagnostic records")
    hashes = {} if source_sha256 is None else {
        str(path): str(digest) for path, digest in source_sha256.items()}
    root = Path(output_dir).expanduser().resolve() / "failed_cell_images"
    jobs, targets = [], set()
    for failure in invalid_cells:
        failure = _mapping(failure, "failed-cell diagnostic")
        side_text = str(failure["side_channel"])
        time = str(failure["time_point"])
        label = _label(failure["cell_index"])
        matching_sides = [side for side in masks if _side_token(side) == side_text]
        if len(matching_sides) != 1:
            raise ValueError(f"cannot identify diagnostic side channel: {side_text}")
        side = matching_sides[0]
        if time not in masks[side] or time not in images.get(side, {}):
            raise ValueError(f"missing diagnostic mask/images at {side}/{time}")
        raw_mask = np.asarray(masks[side][time])
        if raw_mask.ndim != 2 or raw_mask.dtype.kind not in "bui":
            raise ValueError(f"diagnostic mask must be a 2D integer array at {side}/{time}")
        channel_images = _mapping(images[side][time], "diagnostic channel images")
        detection = info.get("cell_detection_opticalChannel_name")
        if detection is None or not str(detection).strip():
            raise ValueError("missing cell_detection_opticalChannel_name for diagnostic export")
        matches = [name for name in channel_images
                   if str(name).casefold() == str(detection).casefold()]
        if len(matches) != 1:
            raise ValueError(
                f"missing or ambiguous segmentation channel {detection!r} at {side}/{time}")
        selected_channel = matches[0]
        channel_images = {selected_channel: channel_images[selected_channel]}
        time_stem = Path(time).stem
        folder = (f"{_diagnostic_component(fov_name)}_side{_diagnostic_component(side)}_"
                  f"{_diagnostic_component(time_stem)}_cell{label}")
        cell_dir = root / folder
        mask = (raw_mask == label).astype(np.uint8)
        image_paths = {}
        for channel, image in channel_images.items():
            array = np.asarray(image)
            if array.ndim != 2 or array.shape != raw_mask.shape:
                raise ValueError(f"diagnostic mask/image mismatch at {side}/{time}/{channel}")
            filename = f"{folder}_{_diagnostic_component(channel)}.tif"
            if filename.casefold() in {name.casefold() for name in image_paths.values()}:
                raise ValueError(f"diagnostic filenames collide at {side}/{time}")
            image_paths[str(channel)] = filename
        mask_filename = (f"{folder}_{_diagnostic_component(selected_channel)}_masks.tif")
        metadata = {
            "diagnostic_only": True, "fov": str(fov_name),
            "segmentation_channel": str(selected_channel),
            "side_channel": side_text, "time_point": time, "cell_index": label,
            "reason": str(failure.get("reason", "DT reconstruction failed")),
            "geometry_mode": "invalid", "umppx": umppx,
            "source_sha256": hashes,
            "source_mask_dtype": str(raw_mask.dtype),
            "mask_pixel_count": int(np.count_nonzero(mask)), "shape": list(raw_mask.shape),
            "mask_semantics": "uint8 binary mask: 1 where saved label mask equals cell_index; 0 elsewhere",
            "image_semantics": "saved registered cropped segmentation-channel image; no rescaling or preprocessing",
            "image_paths": {channel: str(cell_dir / name)
                            for channel, name in image_paths.items()},
            "mask_path": str(cell_dir / mask_filename),
            "metadata_path": str(cell_dir / "diagnostic.json"),
        }
        local_targets = [cell_dir / mask_filename, cell_dir / "diagnostic.json"]
        local_targets += [cell_dir / name for name in image_paths.values()]
        local_keys = {str(path).casefold() for path in local_targets}
        if local_keys & targets:
            raise ValueError(f"duplicate failed-cell diagnostic target: {cell_dir}")
        targets.update(local_keys)
        jobs.append((cell_dir, mask, channel_images, image_paths, metadata, local_targets))
    obsolete_files = {}
    for cell_dir, _, _, _, _, local_targets in jobs:
        old_metadata_path = cell_dir / "diagnostic.json"
        obsolete_files[cell_dir] = []
        if overwrite and old_metadata_path.exists():
            old_metadata = json.loads(old_metadata_path.read_text(encoding="utf-8"))
            if old_metadata.get("diagnostic_only") is not True:
                raise ValueError(f"refusing to replace unknown diagnostic metadata: {old_metadata_path}")
            old_paths = list(_mapping(old_metadata.get("image_paths"),
                                      "previous diagnostic image paths").values())
            old_paths.append(old_metadata["mask_path"])
            current_paths = {str(path.resolve()).casefold() for path in local_targets}
            for old_path in old_paths:
                path = Path(old_path)
                if (not path.is_absolute() or path.resolve().parent != cell_dir.resolve()
                        or path.suffix.casefold() not in {".tif", ".tiff"} or path.is_symlink()):
                    raise ValueError(f"previous diagnostic path is outside its cell folder: {path}")
                if str(path.resolve()).casefold() not in current_paths:
                    obsolete_files[cell_dir].append(path)
        for path in local_targets:
            if path.exists() and not overwrite:
                raise FileExistsError(f"diagnostic output exists: {path}; use --overwrite")
            if path.exists() and not path.is_file():
                raise ValueError(f"diagnostic destination is not a regular file: {path}")
    written = []
    for cell_dir, mask, channel_images, image_paths, metadata, _ in jobs:
        cell_dir.mkdir(parents=True, exist_ok=True)
        _atomic_tiff(Path(metadata["mask_path"]), mask)
        for channel, filename in image_paths.items():
            _atomic_tiff(cell_dir / filename, np.asarray(channel_images[channel]))
        atomic_json(cell_dir / "diagnostic.json", metadata)
        for obsolete in obsolete_files[cell_dir]:
            # Paths were resolved and restricted to this exact cell folder above.
            obsolete.unlink(missing_ok=True)
        written.append(metadata)
    return written


def _csv_keys(frame):
    if not frame.columns.is_unique or any(c not in frame for c in IDENTITY + ["time_s"]):
        raise ValueError("CSV needs unique columns and channels,time_point,cell_index,time_s")
    keys = [_key(s, t, c) for s, t, c in frame[IDENTITY].itertuples(index=False, name=None)]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate CSV cell identities")
    return keys


def normalize_channels(channels):
    if channels is None:
        return None
    if isinstance(channels, str):
        channels = [channels]
    names = []
    for entry in channels:
        for name in re.split(r"[,\s]+", str(entry).strip()):
            if name and name.lower() not in [n.lower() for n in names]:
                names.append(name)
    if not names:
        raise ValueError("fluorescence channel selection is empty")
    return names


def _profile_settings(profile):
    if not profile:
        return None
    _mapping(profile, "saved fluorescence profile")
    positions = np.asarray(profile.get("rel_pos"))
    if positions.ndim != 1 or not len(positions) or not np.all(np.isfinite(positions)):
        raise ValueError("saved profile lacks valid rel_pos; specify fresh profile settings")
    channels = [str(k)[len("median_"):] for k in profile if str(k).startswith("median_")]
    if not channels:
        raise ValueError("cannot infer saved profile channels; specify fresh profile settings")
    return channels, len(positions)


def _resolve_profile(old, companion, images, detection, enabled, channels, bins):
    saved = [p for p in (old, companion) if p]
    if not enabled and channels is None and bins is None and not saved:
        return None
    settings = []
    for profile in saved:
        if channels is not None and bins is not None:
            continue
        settings.append(_profile_settings(profile))
    inferred_channels = [c for c, _ in settings]
    inferred_bins = [b for _, b in settings]
    if channels is None and len({tuple(c) for c in inferred_channels}) > 1:
        raise ValueError("conflicting embedded/companion profile channels; set --flu_distribution_channels")
    if bins is None and len(set(inferred_bins)) > 1:
        raise ValueError("conflicting embedded/companion profile bins; set --flu_distribution_bins")
    requested = channels or (inferred_channels[0] if inferred_channels else [
        name for name in images if str(name).lower() != str(detection).lower()])
    n_bins = bins or (inferred_bins[0] if inferred_bins else 15)
    available = {str(name).lower(): name for name in images}
    selected = []
    for name in requested:
        matched = available.get(str(name).lower())
        # Production skips unavailable requested channels at sparse acquisitions.
        if matched is not None:
            selected.append(matched)
    return selected, n_bins


def _task_compute(mask, images, background, labels, time, umppx, pole_refinement="contour"):
    cells = dt_timepoint_compute(mask, images, background, 4, None, 15, time,
                                umppx=umppx, cell_indices=labels,
                                pole_refinement=pole_refinement)
    # Scalar mask measurements remain defined when geometric reconstruction fails.
    # Profiles still require a valid axis and remain absent for invalid cells.
    for cell in cells:
        if not cell.cellSkeletonSuccess:
            current_mask = np.asarray(mask) == cell.cell_index
            cell._mask_indices = np.nonzero(current_mask)
            cell._mask_shape = tuple(current_mask.shape)
            cell._mask_area_px = float(np.count_nonzero(current_mask))
            cell.mask_area = cell._mask_area_px * cell.umppx ** 2
            cell.channel_imgs = images
            cell.background_fluorescence = background
            cell._compute_fluorescence()
    return cells


def _merge_parameters(old, cell):
    result = dict(old)
    result.update(cell.export_cell_parameters())
    # Refresh known historical derived aliases instead of leaving stale geometry.
    aliases = {
        "centerline": cell.centerline, "spine": cell.spine, "skeleton": cell.skeleton,
        "_curvature": cell._curvature, "_max_width_px": cell._max_width_px,
        "_n_ribs": cell._n_ribs, "_pole_radius": cell._pole_radius,
        "_endpoint_radii": cell._endpoint_radii, "_width_px": cell._width_px,
        "_steplength": getattr(cell, "_steplength", np.array([])),
        "_lengthvector": getattr(cell, "_lengthvector", np.array([])),
        "volume_px": cell._volume_px,
        "body_spine_length_px": cell._spine_length_px,
        "volume_um3": cell._volume_px * cell.umppx ** 3,
        "max_width_um": cell._max_width_px * cell.umppx,
        "pole_radius_um": cell._pole_radius * cell.umppx,
    }
    for name, value in aliases.items():
        if name in old:
            result[name] = value
    return result


def _align_csv(fresh, previous, selection, upgrade_csv, saved_keys, report):
    if previous is None:
        return fresh
    previous = previous.copy(deep=True)
    old_keys, new_keys = _csv_keys(previous), _csv_keys(fresh)
    if saved_keys and set(old_keys) != set(saved_keys):
        raise ValueError("CSV identities do not match saved cell records")
    lookup = {key: i for i, key in enumerate(old_keys)}
    new_lookup = {key: i for i, key in enumerate(new_keys)}
    if selection == "existing":
        if set(old_keys) != set(new_keys):
            raise ValueError("CSV identities do not match recalculated records")
        order = old_keys
    else:
        order = new_keys
        report["counts"]["rows_added"] = len(set(new_keys) - set(old_keys))
        report["counts"]["rows_removed"] = len(set(old_keys) - set(new_keys))
    columns = list(previous.columns)
    if upgrade_csv:
        columns += [name for name in fresh.columns if name not in columns]
    rows = []
    for key in order:
        old_row = previous.iloc[lookup[key]] if key in lookup else None
        new_row = fresh.iloc[new_lookup[key]]
        if old_row is not None:
            old_time = pd.to_numeric(pd.Series([old_row["time_s"]]), errors="coerce").iloc[0]
            if not np.isfinite(old_time) or not np.isclose(old_time, new_row["time_s"], rtol=0, atol=1e-6):
                raise ValueError(f"CSV time_s disagrees with saved timestamp for {key}")
        row = {}
        for name in columns:
            if old_row is not None and name in IDENTITY + ["time_s"]:
                row[name] = old_row[name]
            elif name in fresh:
                row[name] = new_row[name]
            elif old_row is not None:
                row[name] = old_row[name]
            else:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def recompute_export(data, existing_csv=None, *, umppx=0.065, jobs=1,
                     selection="existing", max_cell_number=4, upgrade_csv=False,
                     flu_distribution=False, flu_distribution_channels=None,
                     flu_distribution_bins=None, companion_distribution=None,
                     allow_invalid=False, pole_refinement="contour"):
    """Return (updated PKL mapping, CSV frame, optional profiles, diagnostics).

    Inputs are never mutated. Missing inputs, ambiguous identities, and invalid
    geometry raise ValueError. allow_invalid retains production invalid records
    with explicit diagnostics. Scalar/profile formulas are shared with production.
    The channel contour pipeline saves the initial valid contour reference before
    body midpoint updates and freezes its endpoint radii. Body refinement fixes
    all seed terminals: at most 15 iterations, beta=0.45 and mean matched shift
    <0.25 px, with coverage and endpoint stability required. Rods use one initial
    median body halfwidth; branches use independent initial terminal 20% medians
    (3 to 21 ribs per end). Both fit pole centers within saved local bounds against
    frozen contour arcs. Final strict validation, length bounds and tip coverage
    must pass; any final failure restores the exact initial reference. Mesh
    measurements describe the selected primary body; raw area and fluorescence
    retain the whole original mask.
    pole_refinement="none" is deprecated and cannot disable this pipeline.
    """
    umppx, jobs = _scale(umppx), _positive_int(jobs, "jobs")
    morphology = describe_dt_morphology(pole_refinement)
    pole_refinement = morphology["pole_refinement"]
    max_cell_number = _positive_int(max_cell_number, "max_cell_number")
    if selection not in {"existing", "pipeline", "all"}:
        raise ValueError("selection must be existing, pipeline, or all")
    channels = normalize_channels(flu_distribution_channels)
    bins = None if flu_distribution_bins is None else _positive_int(flu_distribution_bins, "flu_distribution_bins")
    data = _mapping(data, "export")
    info = _mapping(data.get("info_dict"), "info_dict")
    payload = _mapping(data.get("cells_sideChannel_image_dict"), "cells_sideChannel_image_dict")
    if str(info.get("algorithm", "")).upper() != "DT":
        raise ValueError("only DT exports are supported (info_dict.algorithm must be DT)")
    for name in ("time_points", "opticalChannels", "channel_keys"):
        if not isinstance(info.get(name), (list, tuple)):
            raise ValueError(f"missing or invalid info_dict.{name}")
        if len(set(info[name])) != len(info[name]):
            raise ValueError(f"duplicate info_dict.{name}")
    detection = info.get("cell_detection_opticalChannel_name")
    if detection is None:
        raise ValueError("missing cell_detection_opticalChannel_name")
    masks = _mapping(payload.get("fov_cells_mask_dict"), "fov_cells_mask_dict")
    images = _mapping(payload.get("fov_channels_image_dict"), "fov_channels_image_dict")
    seconds = _mapping(payload.get("time_points_seconds"), "time_points_seconds")
    backgrounds = _mapping(payload.get("images_background_fluorescence"), "images_background_fluorescence")
    saved = payload.get("fov_cells_pars_dict", {})
    _mapping(saved, "fov_cells_pars_dict")
    companion = {} if companion_distribution is None else _mapping(companion_distribution, "companion_distribution")
    old_lookup, saved_keys = {}, []
    for side, times in saved.items():
        for time, records in _mapping(times, "saved side-channel times").items():
            if not isinstance(records, (list, tuple)):
                raise ValueError("saved cells must be a list")
            for record in records:
                _mapping(record, "saved cell parameters")
                identity = _key(side, time, record.get("cell_index"))
                if identity in old_lookup:
                    raise ValueError(f"duplicate saved cell identity: {identity}")
                old_lookup[identity] = record
                saved_keys.append(identity)
    if selection == "existing" and not saved_keys and any(
            m is not None and np.any(m) for times in masks.values() for m in times.values()):
        raise ValueError("no saved cell records; use --selection pipeline or --all_cells to rebuild from masks")
    report = {"status": "ok", "parameters": {
        "umppx": umppx, "jobs": jobs, "selection": selection,
        "max_cell_number": max_cell_number, "upgrade_csv": bool(upgrade_csv),
        "flu_distribution": bool(flu_distribution), "flu_distribution_channels": channels,
        "flu_distribution_bins": bins, "allow_invalid": bool(allow_invalid)},
        "morphology": {**morphology, "umppx": umppx},
        "counts": {"side_channels": 0, "time_points": 0, "cells_seen": 0,
                   "cells_computed": 0, "cells_invalid": 0, "cells_recovered": 0,
                   "profiles": 0, "rows_added": 0, "rows_removed": 0},
        "warnings": [], "invalid_cells": []}
    report["parameters"]["pole_refinement"] = pole_refinement
    if any(float(p.get("umppx", umppx)) != umppx for p in old_lookup.values()):
        report["warnings"].append("Requested umppx differs from saved calibration; physical measurements rescaled.")
    time_set = set(info["time_points"])
    tasks, contexts = [], []
    traversal = saved if selection == "existing" else masks
    cells_by_side = {}
    for side, times in traversal.items():
        cells_by_side[side] = {}
        time_order = list(times) if selection == "existing" else list(info["time_points"])
        for time in time_order:
            if time not in time_set:
                raise ValueError(f"time {time!r} absent from info_dict.time_points")
            records = saved.get(side, {}).get(time, [])
            raw_mask = masks.get(side, {}).get(time)
            if raw_mask is None:
                if records:
                    raise ValueError(f"missing mask at {side}/{time}")
                cells_by_side[side][time] = []
                continue
            # Reserve the slot before parallel work so empty times cannot move
            # ahead of earlier nonempty times in saved insertion order.
            cells_by_side[side][time] = []
            mask = np.asarray(raw_mask)
            if mask.ndim != 2 or not mask.size or mask.dtype.kind not in "bui" or np.any(mask < 0):
                raise ValueError(f"mask must be a nonnegative 2D integer array at {side}/{time}")
            if selection == "existing":
                labels = [_label(p["cell_index"]) for p in records]
            elif selection == "pipeline":
                labels = list(range(1, min(int(mask.max()), max_cell_number) + 1))
            else:
                labels = [int(label) for label in np.unique(mask) if label > 0]
            if not labels:
                continue
            if time not in seconds or not np.isfinite(float(seconds[time])):
                raise ValueError(f"missing or invalid timestamp for {time}")
            c_images = _mapping(images.get(side, {}).get(time), f"channel images at {side}/{time}")
            if not c_images:
                raise ValueError(f"missing channel images at {side}/{time}")
            for name, image in c_images.items():
                if np.ndim(image) != 2 or np.shape(image) != mask.shape:
                    raise ValueError(f"mask/image shape mismatch at {side}/{time}/{name}")
                if name not in info["opticalChannels"]:
                    raise ValueError(f"image channel {name!r} absent from opticalChannels")
            background = _mapping(backgrounds.get(time), f"background mapping for {time}")
            for name in c_images:
                value = background.get(name)
                mean = value.get("mean") if isinstance(value, Mapping) else value
                if mean is None:
                    report["warnings"].append(f"Missing background mean at {time}/{name}; production zero fallback applied.")
            # Scalar-only old backgrounds are accepted by CellDT; standard CSV
            # background columns require mean/median dictionaries.
            task = (mask, c_images, background, labels, time, umppx, pole_refinement)
            tasks.append(task)
            contexts.append((side, time, mask, c_images, labels))
        report["counts"]["side_channels"] += 1
    results = Parallel(n_jobs=jobs, backend="loky", return_as="generator", pre_dispatch=jobs)(
        delayed(_task_compute)(*task) for task in tasks)
    for (side, time, mask, c_images, labels), cells in zip(contexts, results):
        cells_by_side[side][time] = cells
        report["counts"]["time_points"] += 1
        for cell in cells:
            identity = _key(side, time, cell.cell_index)
            old = old_lookup.get(identity, {})
            report["counts"]["cells_seen"] += 1
            if not cell.cellSkeletonSuccess:
                report["counts"]["cells_invalid"] += 1
                invalid = {"side_channel": str(side), "time_point": str(time),
                    "cell_index": int(cell.cell_index), "reason": "DT reconstruction failed" if np.any(mask == cell.cell_index) else "label absent from mask"}
                diagnostics = cell.pole_refinement_diagnostics
                invalid.update(pole_refinement=cell.pole_refinement,
                               pole_refinement_diagnostics=diagnostics)
                if np.any(mask == cell.cell_index) and diagnostics.get("reason"):
                    invalid["reason"] = diagnostics["reason"]
                report["invalid_cells"].append(invalid)
                continue
            if (cell.mesh is None or not np.all(np.isfinite(cell.mesh)) or
                    not np.isfinite(cell.spine_length) or cell.spine_length <= 0):
                raise ValueError(f"invalid finite geometry for {identity}")
            report["counts"]["cells_computed"] += 1
            if old and not old.get("cellSkeletonSuccess", False):
                report["counts"]["cells_recovered"] += 1
            old_profile = old.get("flu_distribution", {})
            companion_profile = companion.get(side, {}).get(time, {}).get(cell.cell_index, {})
            settings = _resolve_profile(old_profile, companion_profile, c_images, detection,
                                        flu_distribution, channels, bins)
            if settings:
                selected, n_bins = settings
                cell.compute_flu_distribution(selected, n_bins, mask == cell.cell_index)
                if cell.flu_distribution:
                    report["counts"]["profiles"] += 1
            count = int(np.count_nonzero(mask == cell.cell_index))
            for name, values in cell.flu_level.items():
                if values["pixel_count"] != count:
                    raise ValueError(f"fluorescence mask-pixel count mismatch for {identity}/{name}")
            if cell.flu_distribution and int(cell.flu_distribution["pixel_count"].sum()) != count:
                raise ValueError(f"profile mask-pixel count mismatch for {identity}")
    if report["counts"]["cells_invalid"]:
        report["status"] = "partial"
        if not allow_invalid:
            raise RecalculationError(
                f"{report['counts']['cells_invalid']} invalid cell(s); originals retained. "
                "Use --allow_invalid only to retain flagged production invalid records.", report)
    # Normalize only a temporary view for the production CSV exporter, leaving
    # the original saved background dictionary untouched.
    csv_backgrounds = {}
    for time, values in backgrounds.items():
        csv_backgrounds[time] = {}
        for name, value in _mapping(values, f"background mapping for {time}").items():
            csv_backgrounds[time][name] = value if isinstance(value, Mapping) else {"mean": value, "median": None}
    fresh = export_cells_to_dataframe_dt(cells_by_side, seconds, info["opticalChannels"], csv_backgrounds)
    frame = _align_csv(fresh, existing_csv, selection, upgrade_csv, saved_keys, report)
    updated = dict(data)
    old_morphology = info.get("morphology", {})
    if not isinstance(old_morphology, Mapping):
        raise ValueError("info_dict.morphology must be a mapping when present")
    recalculation = dict(info.get("cell_parameter_recalculation", {}))
    if "cell_parameter_recalculation" in info:
        recalculation["previous_recalculation"] = info["cell_parameter_recalculation"]
    recalculation.update({"method_version": METHOD_VERSION, **report["parameters"]})
    current_morphology = dict(old_morphology)
    if old_morphology:
        recalculation["previous_morphology"] = dict(old_morphology)
    updated["info_dict"] = {**info, "algorithm": "DT",
                            "morphology": {**current_morphology, **report["morphology"]},
                            "cell_parameter_recalculation": recalculation}
    pars = {side: {time: [_merge_parameters(old_lookup.get(_key(side, time, c.cell_index), {}), c)
                          for c in cells] for time, cells in times.items()}
            for side, times in cells_by_side.items()}
    updated["cells_sideChannel_image_dict"] = {**payload, "fov_cells_pars_dict": pars}
    want_distribution = companion_distribution is not None or flu_distribution or channels is not None or bins is not None or report["counts"]["profiles"] > 0
    distribution = collect_flu_distribution_dt(cells_by_side) if want_distribution else None
    return updated, frame, distribution, report


def _read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(set(header)) != len(header):
        raise ValueError(f"duplicate CSV columns in {path}")
    # One inference pass avoids spurious mixed-type warnings on large CSVs
    # containing legitimate blank fluorescence values for failed cells.
    return pd.read_csv(path, dtype={"channels": str, "time_point": str, "cell_index": str},
                       keep_default_na=False, low_memory=False)


def _module_source(modname):
    spec = importlib.util.find_spec(modname)
    return Path(spec.origin) if spec and spec.origin else None


def _provenance(pole_refinement="contour"):
    packages = {}
    for name in ("numpy", "scipy", "numba", "pandas", "joblib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    files = [Path(__file__),
             _module_source("moma_pipeline.io.transactions"),
             _module_source("moma_pipeline.morphology.worker"),
             _module_source("moma_pipeline.io.dt_exports"),
             _module_source("moma_pipeline.morphology.celldt"),
             _module_source("moma_pipeline.morphology.voronoi_dt"),
             _module_source("moma_pipeline.morphology.kernels")]
    morphology = describe_dt_morphology(pole_refinement)
    return {"method_version": METHOD_VERSION, "command": sys.argv, "python": sys.version,
            "executable": sys.executable, "packages": packages,
            "morphology": morphology,
            "code_sha256": {str(p): sha256(p) for p in files}}


def _recover_fov_journals(journal, dry_run):
    """Recover a later interrupted overwrite before reading or publishing pairs."""
    pattern = re.compile(re.escape(journal.stem) + r"(?:_[0-9a-f]{32})?\.json\Z")
    candidates = [p for p in journal.parent.glob(journal.stem + "*.json")
                  if pattern.fullmatch(p.name)]
    incomplete = []
    for candidate in candidates:
        state = json.loads(candidate.read_text(encoding="utf-8"))["state"]
        if state not in {"preparing", "committing", "complete", "rolled_back"}:
            raise ValueError(f"invalid output transaction state: {candidate}")
        if state in {"preparing", "committing"}:
            incomplete.append(candidate)
    if len(incomplete) > 1:
        raise ValueError("multiple incomplete output transactions; manual recovery required")
    if incomplete and dry_run:
        raise ValueError("incomplete output transaction; rerun without --dry_run to recover")
    return [{"journal": str(p), "state": recover_transaction(p)} for p in incomplete]


def analyze_folder(pkl_dir, output_dir=None, *, in_place=False, umppx=0.065, jobs=1,
                   selection="existing", max_cell_number=4, upgrade_csv=False,
                   flu_distribution=False, flu_distribution_channels=None,
                   flu_distribution_bins=None, allow_invalid=False,
                   export_failed_images=False,
                   overwrite=False, skip_existing=False, dry_run=False,
                   pole_refinement="contour"):
    """Process one FOV at a time; publish complete validated output sets only."""
    umppx, jobs = _scale(umppx), _positive_int(jobs, "jobs")
    morphology = describe_dt_morphology(pole_refinement)
    pole_refinement = morphology["pole_refinement"]
    if overwrite and skip_existing:
        raise ValueError("overwrite and skip_existing are mutually exclusive")
    source_dir = Path(pkl_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir else (
        source_dir if in_place else source_dir / "recalculated_data")
    if in_place and destination != source_dir:
        raise ValueError("--in_place cannot be combined with another output directory")
    if destination == source_dir and not in_place:
        raise ValueError("use --in_place to replace source files")
    sources = sorted(source_dir.glob(f"*{SOURCE_SUFFIX}"), key=lambda p: p.name) if source_dir.is_dir() else []
    batch = {"status": "ok", "started_at": datetime.now(timezone.utc).isoformat(),
             "pkl_dir": str(source_dir), "output_dir": str(destination),
             "dry_run": bool(dry_run), "export_failed_images": bool(export_failed_images),
             "provenance": _provenance(pole_refinement), "files": [],
             "counts": {"files_seen": len(sources), "files_written": 0,
                        "files_failed": 0, "files_skipped": 0, "cells_computed": 0,
                        "cells_invalid": 0, "cells_recovered": 0}}
    if not source_dir.is_dir() or not sources:
        batch["status"] = "error"
        batch["error"] = (f"input directory does not exist: {source_dir}" if not source_dir.is_dir()
                          else f"no *{SOURCE_SUFFIX} files found in {source_dir}")
        batch["finished_at"] = datetime.now(timezone.utc).isoformat()
        return batch
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        fov = source.name[:-len(SOURCE_SUFFIX)]
        csv_path = source.with_name(f"{fov}_cells_df.csv")
        profile_path = source.with_name(f"{fov}_flu_distribution.pkl")
        journal = destination / f".{fov}_cell_parameters_transaction.json"
        summary = {"fov": fov, "source_path": str(source), "status": "error", "written": False}
        batch["files"].append(summary)
        diagnostics_attempted = False
        try:
            with ExitStack() as update_lock:
                if not dry_run:
                    update_lock.enter_context(transaction_lock(
                        journal.with_name(f".{fov}_cell_parameters_update")))
                summary["recovered_transactions"] = _recover_fov_journals(journal, dry_run)
                if journal.exists():
                    if dry_run:
                        state = json.loads(journal.read_text(encoding="utf-8"))["state"]
                        if state not in {"complete", "rolled_back"}:
                            raise ValueError("incomplete output transaction; rerun without --dry_run to recover")
                    else:
                        summary["recovery"] = recover_transaction(journal)
                        if summary["recovery"] in {"complete", "rolled_back"}:
                            journal = journal.with_name(
                                f"{journal.stem}_{uuid.uuid4().hex}{journal.suffix}")
                output_pkl, output_csv = destination / source.name, destination / csv_path.name
                output_profile = destination / profile_path.name
                output_summary = destination / f"{fov}_cell_parameters_summary.json"
                if not in_place and any(p.exists() for p in (output_pkl, output_csv, output_profile, output_summary)):
                    if skip_existing:
                        summary["status"] = "skipped"
                        batch["counts"]["files_skipped"] += 1
                        continue
                    if not overwrite:
                        raise FileExistsError("output exists; select --overwrite or --skip_existing")
                inputs = [p for p in (source, csv_path, profile_path) if p.exists()]
                expected = {p: sha256(p) for p in inputs}
                summary["source_sha256"] = {str(p): digest for p, digest in expected.items()}
                with source.open("rb") as stream:
                    data = pickle.load(stream)
                previous = _read_csv(csv_path) if csv_path.exists() else None
                companion = None
                if profile_path.exists():
                    with profile_path.open("rb") as stream:
                        companion = pickle.load(stream)
                updated, frame, distribution, report = recompute_export(data, previous,
                    umppx=umppx, jobs=jobs, selection=selection, max_cell_number=max_cell_number,
                    upgrade_csv=upgrade_csv, flu_distribution=flu_distribution,
                    flu_distribution_channels=flu_distribution_channels,
                    flu_distribution_bins=flu_distribution_bins,
                    companion_distribution=companion, allow_invalid=allow_invalid,
                    pole_refinement=pole_refinement)
                summary.update(report)
                if export_failed_images:
                    if dry_run:
                        summary["failed_cell_images_planned"] = len(report["invalid_cells"])
                    elif report["invalid_cells"]:
                        diagnostics_attempted = True
                        summary["failed_cell_images"] = export_failed_cell_images(
                            updated, report["invalid_cells"], destination, fov,
                            overwrite=overwrite or in_place,
                            source_sha256=expected, umppx=umppx)
                    else:
                        summary["failed_cell_images"] = []
                summary["rows"] = len(frame)
                summary["columns"] = list(frame.columns)
                for name in ("cells_computed", "cells_invalid", "cells_recovered"):
                    batch["counts"][name] += report["counts"][name]
                # Validate bytes before a transaction can replace any destination.
                files = {output_pkl: pickle.dumps(updated, protocol=pickle.HIGHEST_PROTOCOL),
                         output_csv: frame.to_csv(index=False).encode("utf-8")}
                pickle.loads(files[output_pkl])
                staged_csv = pd.read_csv(io.BytesIO(files[output_csv]),
                                         dtype={"channels": str,"time_point": str,"cell_index": str})
                if _csv_keys(staged_csv) != _csv_keys(frame) or list(staged_csv.columns) != list(frame.columns):
                    raise ValueError("staged CSV identity/schema validation failed")
                if distribution is not None:
                    files[output_profile] = pickle.dumps(distribution, protocol=pickle.HIGHEST_PROTOCOL)
                    profile_summary = destination / f"{fov}_flu_distribution_summary.json"
                    files[profile_summary] = (json.dumps({"status": report["status"],
                        "source_path": str(output_pkl), "source_size_bytes": len(files[output_pkl]),
                        "source_sha256": hashlib.sha256(files[output_pkl]).hexdigest(),
                        "generated_by": "cell_parameters_from_pkls", "parameters": report["parameters"],
                        "profiles": report["counts"]["profiles"]}, indent=2) + "\n").encode()
                # Refuse to strand old destination profiles after new geometry.
                elif output_profile.exists():
                    raise ValueError("destination has profiles but source has none; use a clean output directory or enable profiles")
                summary["output_sha256"] = {str(p): hashlib.sha256(b).hexdigest() for p, b in files.items()}
                summary["provenance"] = batch["provenance"]
                summary["written"] = not dry_run
                for path, digest in expected.items():
                    if sha256(path) != digest:
                        raise ValueError(f"source changed during calculation: {path}")
                if not dry_run:
                    files[output_summary] = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
                    summary["transaction"] = publish_files(
                        files, journal,
                        expected_sources=expected if in_place else None)
                    summary["backup_dir"] = summary["transaction"].get("backup_dir")
                    batch["counts"]["files_written"] += 1
    
        except Exception as exc:
            if getattr(exc, "report", None) is not None:
                summary.update(exc.report)
                for name in ("cells_computed", "cells_invalid", "cells_recovered"):
                    batch["counts"][name] += exc.report["counts"][name]
            if (export_failed_images and summary.get("invalid_cells")
                    and not diagnostics_attempted):
                diagnostics_attempted = True
                if dry_run:
                    summary["failed_cell_images_planned"] = len(summary["invalid_cells"])
                else:
                    try:
                        summary["failed_cell_images"] = export_failed_cell_images(
                            data, summary["invalid_cells"], destination, fov,
                            overwrite=overwrite or in_place,
                            source_sha256=expected, umppx=umppx)
                    except Exception as diagnostic_exc:
                        summary["diagnostic_error"] = (
                            f"{type(diagnostic_exc).__name__}: {diagnostic_exc}")
            summary.update(status="error", written=False, error=f"{type(exc).__name__}: {exc}")
            batch["counts"]["files_failed"] += 1
            if not dry_run:
                # Separate failure file never replaces a prior successful summary.
                atomic_json(destination / f"{fov}_cell_parameters_error.json", summary)
        print(f"{fov}: {summary['status']}" + (f" ({summary['rows']} rows)" if "rows" in summary else ""), flush=True)
    batch["status"] = "error" if batch["counts"]["files_failed"] else (
        "partial" if batch["counts"]["cells_invalid"] else "ok")
    batch["finished_at"] = datetime.now(timezone.utc).isoformat()
    if not dry_run:
        atomic_json(destination / "cell_parameters_run_summary.json", batch)
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl_dir", "--pkl-dir", required=True)
    parser.add_argument("--output_dir", "--output-dir", help="default: <pkl_dir>/recalculated_data")
    parser.add_argument("--in_place", "--in-place", action="store_true", help="replace source pairs, retaining recoverable backups")
    parser.add_argument("--umppx", "--um-per-pixel", type=float, default=0.065, help="micrometers per pixel (default: 0.065)")
    parser.add_argument("--jobs", type=int, default=1, help="positive number of CPU workers within one FOV")
    parser.add_argument("--pole_refinement", "--pole-refinement", choices=["none", "contour"],
                        default="contour", help="channel contour pipeline with branch baseline poles (default: contour); none is a deprecated equivalent")
    population = parser.add_mutually_exclusive_group()
    population.add_argument("--selection", choices=["existing", "pipeline", "all"], default="existing")
    population.add_argument("--all_cells", "--all-cells", action="store_true", help="calculate every present mask label")
    parser.add_argument("--max_cell_number", "--max-cell-number", type=int, default=4, help="cell cap for --selection pipeline")
    parser.add_argument("--upgrade_csv", "--upgrade-csv", action="store_true", help="add missing current production columns")
    parser.add_argument("--flu_distribution", "--flu-distribution", action="store_true", help="enable new profiles; existing profiles always refreshed")
    parser.add_argument("--flu_distribution_channels", "--flu-distribution-channels", nargs="+")
    parser.add_argument("--flu_distribution_bins", "--flu-distribution-bins", type=int, help="new profile default: 15; existing default: infer saved bins")
    parser.add_argument("--allow_invalid", "--allow-invalid", action="store_true", help="retain explicitly flagged failed cells instead of blocking that FOV")
    parser.add_argument("--export_failed_images", "--export-failed-images", action="store_true",
                        help="write each failed cell's segmentation-channel TIFF and matching _masks.tif, even when publication is blocked")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--overwrite", action="store_true", help="replace existing destination outputs with backups")
    policy.add_argument("--skip_existing", "--skip-existing", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="calculate and validate without writing any files")
    args = vars(parser.parse_args(argv))
    if args.pop("all_cells"):
        args["selection"] = "all"
    try:
        result = analyze_folder(**args)
    except Exception as exc:
        print(f"Recalculation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("BEGIN_BATCH_JSON")
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "output_dir": result["output_dir"], "umppx": args["umppx"]}, indent=2))
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
