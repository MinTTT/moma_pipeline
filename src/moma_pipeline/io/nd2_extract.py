"""Extract a Nikon ND2 time-lapse into the MoMA per-FOV TIFF folder layout.

Output layout (data-structure version 0.2, consumed unchanged by
``Moma_CPM_FOV_DT_pipeline.run_pipeline``)::

    save_dir/
      nd2_extraction_manifest.json
      fov_<p>/<channel>/fov_<p>_<channel>_t_<t>.tif

Every TIFF carries shaped metadata:

* ``acq_time``      -- ``"YYYY-mm-dd-HH:MM:SS,<unix_epoch_seconds>"``, the same
  string convention as FOV-folder TIFFs, so ``utilities.read_image`` returns
  Unix-epoch seconds for ``time_s``.
* ``acq_time_jdn``  -- the raw ND2 ``absoluteJulianDayNumber`` (days).
* ``channel`` / ``nd2_channel_name`` -- folder channel name / original ND2 name.

Design notes
------------
* Frames are mapped to (time, position) through ``ND2File.loop_indices`` rather
  than an assumed loop order; loops other than T/P with more than one element
  (e.g. Z stacks) are rejected.
* Each TIFF is written to ``*.tmp`` and atomically renamed, and the manifest is
  rewritten after every completed time point, so an interrupted extraction can
  be resumed without half-written files.
* This module deliberately imports neither torch, cellpose nor cv2.

Ported from Moma_cellPose.nd2_extract into moma_pipeline.io.nd2_extract;
the legacy ``utilities`` import is now a function-level
``moma_pipeline.io.images.read_image`` so this module imports without cv2.

@author: CHU Pan
@e-mail: pan_chu@outlook.com
"""

import datetime
import json
import os
import re
import warnings

import numpy as np
import tifffile as tiff
from tqdm import tqdm

MANIFEST_NAME = 'nd2_extraction_manifest.json'
MANIFEST_SCHEMA = 1
TIME_ENCODING = 'unix_epoch_seconds_from_nd2_jdn'
JDN_UNIX_EPOCH = 2440587.5          # Julian day number of 1970-01-01T00:00:00
SECONDS_PER_DAY = 86400.0
EXTRACT_MODES = ('auto', 'force', 'skip')

# Characters that break the `fov_<p>_<channel>_t_<t>` name parsing ('_') or
# Windows paths. Spaces are kept for backward compatibility with older
# extractions (they never broke parsing).
_UNSAFE_CHANNEL_CHARS = re.compile(r'[_<>:"/\\|?*\x00-\x1f]+')


# =============================================================================
# Pure helpers
# =============================================================================

def jdn_to_unix_seconds(jdn):
    """Convert a Julian day number (days) to Unix-epoch seconds."""
    return (float(jdn) - JDN_UNIX_EPOCH) * SECONDS_PER_DAY


def format_acq_time(epoch_seconds):
    """Render the FOV-TIFF ``acq_time`` string: ``'<local date>,<epoch>'``.

    Matches the acquisition-software convention (local wall-clock date, then
    Unix-epoch seconds). Only the numeric part after the last comma is parsed
    by ``read_image``; it is written with ``repr`` so the float round-trips.
    """
    stamp = datetime.datetime.fromtimestamp(epoch_seconds)
    return f"{stamp.strftime('%Y-%m-%d-%H:%M:%S')},{float(epoch_seconds)!r}"


def sanitize_channel_names(channel_names):
    """Map ND2 channel names to folder-safe names.

    Underscores and Windows-illegal characters are replaced by '-', trailing
    dots/spaces are stripped. Names that would make the image file name
    ambiguous for ``MomaFOV`` (substrings of the fixed name parts, or pure
    digits) and post-sanitization collisions raise ``ValueError``.

    Returns a list of sanitized names in the input order.
    """
    sanitized = []
    for index, name in enumerate(channel_names):
        safe = _UNSAFE_CHANNEL_CHARS.sub('-', str(name))
        safe = re.sub(r'-{2,}', '-', safe).strip(' .-')
        if not safe:
            safe = f'ch{index}'
        if safe.isdigit() or safe in 'fov' or safe in 'tif':
            raise ValueError(
                f"ND2 channel name '{name}' (folder name '{safe}') would make "
                f"image file names ambiguous; rename the channel in the ND2 "
                f"metadata or extract manually.")
        sanitized.append(safe)

    lowered = [name.lower() for name in sanitized]
    if len(set(lowered)) != len(lowered):
        raise ValueError(
            f'ND2 channel names collide after sanitization: '
            f'{list(channel_names)} -> {sanitized}')
    return sanitized


def image_file_name(fov_index, channel, time_index):
    """File name of one extracted image (data-structure version 0.2)."""
    return f'fov_{fov_index}_{channel}_t_{time_index}.tif'


def build_frame_map(sizes, loop_indices):
    """Map (time_index, position_index) -> ND2 frame index.

    Returns ``(frame_map, time_number, fov_number)``. Raises ``ValueError`` for
    unsupported loops (anything other than T/P with more than one element) or
    when the loop indices do not cover every (T, P) pair exactly once.
    """
    unsupported = [axis for axis, size in sizes.items()
                   if axis not in ('T', 'P', 'C', 'Y', 'X') and size > 1]
    if unsupported:
        raise ValueError(
            f'Unsupported ND2 axes {unsupported} (sizes: {dict(sizes)}). '
            f'Only time (T), position (P) and in-frame channels (C) are '
            f'supported; Z stacks / RGB must be projected beforehand.')

    time_number = int(sizes.get('T', 1))
    fov_number = int(sizes.get('P', 1))
    frame_map = {}
    for frame_index, loop in enumerate(loop_indices):
        extra = {k: v for k, v in loop.items() if k not in ('T', 'P') and v}
        if extra:
            raise ValueError(
                f'Frame {frame_index} has unsupported loop indices {loop}.')
        key = (int(loop.get('T', 0)), int(loop.get('P', 0)))
        if key in frame_map:
            raise ValueError(f'Duplicate ND2 frame for (T, P) = {key}.')
        frame_map[key] = frame_index

    expected = time_number * fov_number
    if len(frame_map) != expected:
        raise ValueError(
            f'ND2 loop indices cover {len(frame_map)} (T, P) pairs, expected '
            f'{expected} (T={time_number}, P={fov_number}).')
    return frame_map, time_number, fov_number


def split_channels(frame, channel_count):
    """Split one ND2 frame into a list of 2D channel images."""
    frame = np.asarray(frame)
    if channel_count == 1:
        if frame.ndim != 2:
            raise ValueError(f'Expected a 2D single-channel frame, got shape '
                             f'{frame.shape}.')
        return [frame]
    if frame.ndim != 3 or frame.shape[0] != channel_count:
        raise ValueError(f'Expected frame shape (C={channel_count}, Y, X), '
                         f'got {frame.shape}.')
    return [frame[ch_i] for ch_i in range(channel_count)]


def frame_channel_jdns(metadata, channel_count, frame_index):
    """Per-channel acquisition JDNs from an ND2 frame metadata object."""
    channels = getattr(metadata, 'channels', None) or []
    if len(channels) != channel_count:
        raise ValueError(
            f'Frame {frame_index}: metadata lists {len(channels)} channels, '
            f'expected {channel_count}.')
    jdns = []
    for ch_i, channel in enumerate(channels):
        try:
            jdn = float(channel.time.absoluteJulianDayNumber)
        except (AttributeError, TypeError, ValueError):
            jdn = None
        if jdn is None or not np.isfinite(jdn) or jdn < 1:
            raise ValueError(
                f'Frame {frame_index}, channel {ch_i}: missing absolute '
                f'acquisition time (absoluteJulianDayNumber).')
        jdns.append(jdn)
    return jdns


# =============================================================================
# Manifest
# =============================================================================

def nd2_fingerprint(nd2_path):
    """Cheap identity of an ND2 file (path, size, mtime); no full hash."""
    stat = os.stat(nd2_path)
    return {
        'path': os.path.abspath(nd2_path),
        'size_bytes': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def manifest_path(save_dir):
    return os.path.join(save_dir, MANIFEST_NAME)


def read_manifest(save_dir):
    path = manifest_path(save_dir)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def _write_json_atomic(path, payload):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
    os.replace(tmp_path, path)


def _manifest_identity(manifest):
    """Fields that must match for a manifest to describe the same extraction."""
    return {key: manifest.get(key) for key in
            ('schema', 'nd2', 'sizes', 'channels_nd2', 'channels',
             'time_encoding', 'time_number', 'fov_number')}


def _existing_fov_dirs(save_dir):
    if not os.path.isdir(save_dir):
        return []
    return sorted(d for d in os.listdir(save_dir)
                  if d.startswith('fov') and
                  os.path.isdir(os.path.join(save_dir, d)))


def _probe_legacy_time_unit(save_dir, fov_dirs):
    """Warn when a reused (manifest-less) extraction stores Julian days."""
    from moma_pipeline.io.images import read_image  # local: keeps the module import light
    fov_dir = os.path.join(save_dir, fov_dirs[0])
    for channel in sorted(os.listdir(fov_dir)):
        channel_dir = os.path.join(fov_dir, channel)
        if not os.path.isdir(channel_dir):
            continue
        tifs = sorted(f for f in os.listdir(channel_dir) if f.endswith('.tif'))
        if not tifs:
            continue
        try:
            _, value = read_image(os.path.join(channel_dir, tifs[0]),
                                  read_time=True, time_key='acq_time')
        except Exception as exc:  # metadata missing or unreadable
            return f'could not read acq_time from {tifs[0]} ({exc})'
        if isinstance(value, (int, float)) and 1e6 < value < 1e7:
            return (f'acq_time={value} in {tifs[0]} looks like a Julian day '
                    f'number; time_s would be in days, not seconds')
        return None
    return None


# =============================================================================
# Extraction
# =============================================================================

def _open_nd2(nd2_path):
    import nd2
    return nd2.ND2File(nd2_path)


def extract_nd2_to_fov_tiffs(nd2_path, save_dir, mode='auto', reader=None,
                             log=print):
    """Extract an ND2 file into ``save_dir/fov_<p>/<channel>/`` TIFFs.

    Parameters
    ----------
    nd2_path : str
        Input ND2 file.
    save_dir : str
        Output root; becomes the ``fovs_dir`` of the FOV DT pipeline.
    mode : {'auto', 'force', 'skip'}
        ``auto``  -- reuse a complete matching extraction, resume a partial
        one, extract when nothing exists; refuse to touch manifest-less
        (legacy) FOV folders.
        ``force`` -- re-extract every frame.
        ``skip``  -- never extract; reuse existing FOV folders (warns when no
        complete manifest proves them complete or when legacy Julian-day times
        are detected).
    reader : object, optional
        Pre-opened ND2-like reader (``sizes``, ``loop_indices``,
        ``read_frame``, ``frame_metadata``, ``is_rgb``, ``close``). Used by
        tests; when None the file is opened with ``nd2.ND2File``.
    log : callable
        Message sink (default ``print``).

    Returns
    -------
    dict
        The extraction manifest (also written to ``save_dir``).
    """
    if mode not in EXTRACT_MODES:
        raise ValueError(f'mode must be one of {EXTRACT_MODES}, got {mode!r}')
    if not os.path.exists(nd2_path):
        raise FileNotFoundError(f'ND2 file not found: {nd2_path}')

    existing_manifest = read_manifest(save_dir)
    fov_dirs = _existing_fov_dirs(save_dir)

    if mode == 'skip':
        if not fov_dirs:
            raise FileNotFoundError(
                f'--extract skip requested but no FOV folders exist in '
                f'{save_dir}.')
        if existing_manifest is None or not existing_manifest.get('complete'):
            warnings.warn(
                f'Reusing {len(fov_dirs)} FOV folder(s) in {save_dir} without '
                f'a complete {MANIFEST_NAME}; completeness is unverified.')
            problem = _probe_legacy_time_unit(save_dir, fov_dirs)
            if problem:
                warnings.warn(f'Legacy extraction: {problem}. Re-run with '
                              f'--extract force to write epoch seconds.')
            return existing_manifest or {
                'schema': MANIFEST_SCHEMA, 'complete': None,
                'legacy_reuse': True, 'fov_dirs': fov_dirs}
        log(f'Reusing complete extraction in {save_dir} (--extract skip).')
        return existing_manifest

    own_reader = reader is None
    if own_reader:
        reader = _open_nd2(nd2_path)
    try:
        return _extract(nd2_path, save_dir, mode, reader, existing_manifest,
                        fov_dirs, log)
    finally:
        if own_reader:
            reader.close()


def _extract(nd2_path, save_dir, mode, reader, existing_manifest, fov_dirs,
             log):
    if getattr(reader, 'is_rgb', False):
        raise ValueError('RGB ND2 files are not supported.')
    sizes = {str(k): int(v) for k, v in dict(reader.sizes).items()}
    frame_map, time_number, fov_number = build_frame_map(
        sizes, reader.loop_indices)

    first_meta = reader.frame_metadata(0)
    channels_nd2 = [str(ch.channel.name) for ch in first_meta.channels]
    channel_count = len(channels_nd2)
    if channel_count != int(sizes.get('C', 1)):
        raise ValueError(f'Metadata lists {channel_count} channels but sizes '
                         f'report C={sizes.get("C", 1)}.')
    channels = sanitize_channel_names(channels_nd2)

    manifest = {
        'schema': MANIFEST_SCHEMA,
        'nd2': nd2_fingerprint(nd2_path),
        'sizes': sizes,
        'channels_nd2': channels_nd2,
        'channels': channels,
        'time_encoding': TIME_ENCODING,
        'time_number': time_number,
        'fov_number': fov_number,
        'fov_names': [f'fov_{p}' for p in range(fov_number)],
        'file_name_pattern': 'fov_{p}/{channel}/fov_{p}_{channel}_t_{t}.tif',
        'completed_time_points': 0,
        'complete': False,
    }

    start_time = 0
    if mode == 'auto':
        if existing_manifest is not None:
            if (_manifest_identity(existing_manifest)
                    != _manifest_identity(manifest)):
                raise ValueError(
                    f'{manifest_path(save_dir)} describes a different ND2 file '
                    f'or extraction settings; use --extract force to '
                    f're-extract, or choose another --save_dir.')
            if existing_manifest.get('complete'):
                log(f'Extraction already complete in {save_dir}; reusing it.')
                return existing_manifest
            start_time = int(existing_manifest.get('completed_time_points', 0))
            log(f'Resuming extraction at time point {start_time}/'
                f'{time_number}.')
        elif fov_dirs:
            raise FileExistsError(
                f'{save_dir} already contains {len(fov_dirs)} FOV folder(s) '
                f'but no {MANIFEST_NAME} (legacy or foreign extraction). Use '
                f'--extract force to re-extract (writes epoch-second times), '
                f'or --extract skip to reuse them as-is.')

    if channels != channels_nd2:
        log(f'Channel folder names sanitized: '
            f'{dict(zip(channels_nd2, channels))}')
    log(f'Discovered channels: {channels}')

    os.makedirs(save_dir, exist_ok=True)
    for p in range(fov_number):
        for channel in channels:
            os.makedirs(os.path.join(save_dir, f'fov_{p}', channel),
                        exist_ok=True)

    manifest['completed_time_points'] = start_time
    _write_json_atomic(manifest_path(save_dir), manifest)

    log(f'Extracting {fov_number} FOV(s) x {time_number} time point(s) x '
        f'{channel_count} channel(s), starting at t={start_time}...')
    for t in tqdm(range(start_time, time_number), desc='Extracting ND2',
                  initial=start_time, total=time_number):
        for p in range(fov_number):
            frame_index = frame_map[(t, p)]
            images = split_channels(reader.read_frame(frame_index),
                                    channel_count)
            jdns = frame_channel_jdns(reader.frame_metadata(frame_index),
                                      channel_count, frame_index)
            for ch_i, channel in enumerate(channels):
                epoch_seconds = jdn_to_unix_seconds(jdns[ch_i])
                image_path = os.path.join(save_dir, f'fov_{p}', channel,
                                          image_file_name(p, channel, t))
                tmp_path = image_path + '.tmp'
                tiff.imwrite(tmp_path, images[ch_i], metadata={
                    'channel': channel,
                    'nd2_channel_name': channels_nd2[ch_i],
                    'acq_time': format_acq_time(epoch_seconds),
                    'acq_time_jdn': jdns[ch_i],
                })
                os.replace(tmp_path, image_path)
        manifest['completed_time_points'] = t + 1
        _write_json_atomic(manifest_path(save_dir), manifest)

    manifest['complete'] = True
    _write_json_atomic(manifest_path(save_dir), manifest)
    _warn_unexpected_files(save_dir, channels, time_number, fov_number)
    log(f'Extracted {fov_number} FOV folder(s) to {save_dir}')
    return manifest


def _warn_unexpected_files(save_dir, channels, time_number, fov_number):
    """Warn about stale images/folders the extraction did not write.

    Nothing is deleted: stale files (e.g. from an earlier, longer extraction
    into the same folder) would otherwise be picked up by MomaFOV.
    """
    expected_fovs = {f'fov_{p}' for p in range(fov_number)}
    stale = [d for d in _existing_fov_dirs(save_dir) if d not in expected_fovs]
    for p in range(fov_number):
        fov_dir = os.path.join(save_dir, f'fov_{p}')
        for entry in os.listdir(fov_dir):
            if os.path.isdir(os.path.join(fov_dir, entry)) and \
                    entry not in channels:
                stale.append(f'fov_{p}/{entry}')
        for channel in channels:
            expected = {image_file_name(p, channel, t)
                        for t in range(time_number)}
            channel_dir = os.path.join(fov_dir, channel)
            stale += [f'fov_{p}/{channel}/{name}'
                      for name in os.listdir(channel_dir)
                      if name not in expected]
    if stale:
        preview = ', '.join(stale[:5]) + (' ...' if len(stale) > 5 else '')
        warnings.warn(
            f'{len(stale)} file(s)/folder(s) in {save_dir} were not written '
            f'by this extraction and will be seen by the FOV pipeline: '
            f'{preview}. Remove them or use a clean --save_dir.')


def summarize_manifest(manifest):
    """Compact, JSON-serializable summary for run-parameter provenance."""
    if not manifest:
        return None
    keys = ('nd2', 'sizes', 'channels_nd2', 'channels', 'time_encoding',
            'time_number', 'fov_number', 'completed_time_points', 'complete',
            'legacy_reuse')
    return {key: manifest[key] for key in keys if key in manifest}
