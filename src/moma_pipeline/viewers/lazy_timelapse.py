"""Lazy napari timelapse viewer.

Ported from MomaViewer/lazy_napari_timelapse_viewer.py (self-contained:
napari + qtpy + tifffile + numpy only). Scans a dataset root of
fov_<n>/<channel>/fov_<n>_<channel>_t<k>.tif(f) trees and browses it
lazily frame by frame.
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Any, Generator

import napari
import numpy as np
import tifffile
from napari.qt.threading import create_worker
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


TIFF_PATTERN = re.compile(r"^fov_(\d+)_(.+?)[_-]*t[_-]*(\d+)\.tiff?$", re.IGNORECASE)


@dataclass
class DatasetIndex:
    root: Path
    index: dict[int, dict[str, dict[int, Path]]]
    fovs: list[int]
    channels: list[str]
    sample_shape_by_channel: dict[str, tuple[int, ...]]
    sample_dtype_by_channel: dict[str, np.dtype]


class ImageCache:
    def __init__(self, max_items: int = 100):
        self.max_items = max(0, int(max_items))
        self._items: OrderedDict[tuple[int, str, int], np.ndarray] = OrderedDict()

    def get(self, key: tuple[int, str, int]) -> np.ndarray | None:
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key: tuple[int, str, int], value: np.ndarray) -> None:
        if self.max_items <= 0:
            return
        self._items[key] = value
        self._items.move_to_end(key)
        self._trim()

    def clear(self) -> None:
        self._items.clear()

    def set_max_items(self, n: int) -> None:
        self.max_items = max(0, int(n))
        self._trim()

    def _trim(self) -> None:
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)


class LazyTimelapseWidget(QWidget):
    FULL_TIMELINE = "Full timeline"
    COMPLETE_CHANNELS = "Complete channels"

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.dataset: DatasetIndex | None = None
        self.cache = ImageCache(max_items=100)
        self.layers: dict[str, Any] = {}
        self.channel_controls: dict[str, dict[str, Any]] = {}
        self.timeline: list[int] = []
        self.current_real_t: int | None = None
        self._updating_ui = False
        self._scan_worker = None
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance_playback)

        self._build_ui()
        self._bind_shortcuts()
        self._set_status("Select a root folder, then scan.")

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)

        folder_group = QGroupBox("Folder import")
        folder_layout = QGridLayout(folder_group)
        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("Root folder")
        self.browse_button = QPushButton("Browse")
        self.scan_button = QPushButton("Scan")
        folder_layout.addWidget(self.root_edit, 0, 0, 1, 2)
        folder_layout.addWidget(self.browse_button, 1, 0)
        folder_layout.addWidget(self.scan_button, 1, 1)
        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        folder_layout.addWidget(self.scan_progress, 2, 0, 1, 2)
        root_layout.addWidget(folder_group)

        timeline_group = QGroupBox("Timeline")
        timeline_layout = QFormLayout(timeline_group)
        self.fov_combo = QComboBox()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([self.FULL_TIMELINE, self.COMPLETE_CHANNELS])
        self.reference_combo = QComboBox()
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_spin = QSpinBox()
        self.time_spin.setMinimum(0)
        self.time_spin.setMaximum(0)
        self.play_button = QPushButton("Play")
        self.play_start_spin = QSpinBox()
        self.play_end_spin = QSpinBox()
        for spin in (self.play_start_spin, self.play_end_spin):
            spin.setMinimum(0)
            spin.setMaximum(0)
        self.play_fps_spin = QDoubleSpinBox()
        self.play_fps_spin.setRange(0.1, 120.0)
        self.play_fps_spin.setDecimals(2)
        self.play_fps_spin.setSingleStep(0.5)
        self.play_fps_spin.setValue(5.0)

        play_range_widget = QWidget()
        play_range_layout = QHBoxLayout(play_range_widget)
        play_range_layout.setContentsMargins(0, 0, 0, 0)
        play_range_layout.addWidget(self.play_start_spin)
        play_range_layout.addWidget(QLabel("to"))
        play_range_layout.addWidget(self.play_end_spin)

        self.time_label = QLabel("Frame index: 0 / 0\nReal timepoint: t = -")
        timeline_layout.addRow("FOV", self.fov_combo)
        timeline_layout.addRow("Time mode", self.mode_combo)
        timeline_layout.addRow("Reference", self.reference_combo)
        timeline_layout.addRow("Slider", self.time_slider)
        timeline_layout.addRow("Frame", self.time_spin)
        timeline_layout.addRow("Playback", self.play_button)
        timeline_layout.addRow("Frame range", play_range_widget)
        timeline_layout.addRow("Speed (fps)", self.play_fps_spin)
        timeline_layout.addRow(self.time_label)
        root_layout.addWidget(timeline_group)

        self.channels_group = QGroupBox("Channels")
        channels_group_layout = QVBoxLayout(self.channels_group)
        self.channels_container = QWidget()
        self.channels_layout = QVBoxLayout(self.channels_container)
        self.channels_layout.setContentsMargins(0, 0, 0, 0)
        channels_scroll = QScrollArea()
        channels_scroll.setWidgetResizable(True)
        channels_scroll.setWidget(self.channels_container)
        channels_group_layout.addWidget(channels_scroll)
        root_layout.addWidget(self.channels_group, stretch=1)

        cache_group = QGroupBox("Cache")
        cache_layout = QHBoxLayout(cache_group)
        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(0, 100000)
        self.cache_size_spin.setValue(100)
        self.clear_cache_button = QPushButton("Clear cache")
        cache_layout.addWidget(QLabel("Images"))
        cache_layout.addWidget(self.cache_size_spin)
        cache_layout.addWidget(self.clear_cache_button)
        root_layout.addWidget(cache_group)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label)

        self.browse_button.clicked.connect(self.browse_root)
        self.scan_button.clicked.connect(self.scan_dataset)
        self.fov_combo.currentIndexChanged.connect(self._timeline_control_changed)
        self.mode_combo.currentIndexChanged.connect(self._timeline_control_changed)
        self.reference_combo.currentIndexChanged.connect(self._timeline_control_changed)
        self.time_slider.valueChanged.connect(self._slider_changed)
        self.time_spin.valueChanged.connect(self._spin_changed)
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_start_spin.valueChanged.connect(self._play_range_changed)
        self.play_end_spin.valueChanged.connect(self._play_range_changed)
        self.play_fps_spin.valueChanged.connect(self._play_speed_changed)
        self.cache_size_spin.valueChanged.connect(self.cache.set_max_items)
        self.clear_cache_button.clicked.connect(self._clear_cache)

    def _bind_shortcuts(self) -> None:
        self.viewer.bind_key("Right", self.next_frame, overwrite=True)
        self.viewer.bind_key("D", self.next_frame, overwrite=True)
        self.viewer.bind_key("Left", self.prev_frame, overwrite=True)
        self.viewer.bind_key("A", self.prev_frame, overwrite=True)
        self.viewer.bind_key("Up", self.prev_fov, overwrite=True)
        self.viewer.bind_key("W", self.prev_fov, overwrite=True)
        self.viewer.bind_key("Down", self.next_fov, overwrite=True)
        self.viewer.bind_key("S", self.next_fov, overwrite=True)

    def browse_root(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select root folder", self.root_edit.text())
        if folder:
            self.root_edit.setText(folder)

    def scan_dataset(self) -> None:
        if self._scan_worker is not None:
            return

        root_text = self.root_edit.text().strip()
        if not root_text:
            self._set_status("No root folder selected.")
            return

        root = Path(root_text)
        if not root.exists() or not root.is_dir():
            self._set_status(f"Root folder does not exist: {root}")
            return

        self.stop_playback()
        self.root_edit.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.play_button.setEnabled(False)
        self._update_scan_progress((0, 0, "Finding FOV folders"))
        self.scan_progress.setVisible(True)
        self._set_status(f"Scanning {root}...")

        self._scan_worker = create_worker(self._scan_root, root, _ignore_errors=True)
        self._scan_worker.yielded.connect(self._update_scan_progress)
        self._scan_worker.returned.connect(self._scan_complete)
        self._scan_worker.errored.connect(self._scan_failed)
        self._scan_worker.finished.connect(self._scan_finished)
        self._scan_worker.start()

    def _update_scan_progress(self, progress: tuple[int, int, str]) -> None:
        completed, total, stage = progress
        self.scan_progress.setRange(0, total)
        self.scan_progress.setValue(completed)
        self.scan_progress.setFormat(f"{stage}: %v / %m (%p%)" if total else f"{stage}...")

    def _scan_failed(self, exc: Exception) -> None:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        self.scan_progress.setRange(0, 1)
        self.scan_progress.setValue(0)
        self.scan_progress.setFormat("Scan failed")
        self._set_status(f"Scan failed: {exc}")

    def _scan_finished(self) -> None:
        self._scan_worker = None
        self.root_edit.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.scan_button.setEnabled(True)
        self.play_button.setEnabled(True)

    def _scan_complete(self, dataset: DatasetIndex) -> None:
        self.dataset = dataset
        self.cache.clear()
        self.layers.clear()
        self.channel_controls.clear()
        self.viewer.layers.clear()
        self._populate_dataset_controls()
        self.rebuild_channel_controls()
        self.recompute_timeline()
        n_images = sum(
            len(times)
            for channels in dataset.index.values()
            for times in channels.values()
        )
        self._set_status(
            f"Scanned {len(dataset.fovs)} FOV(s), {len(dataset.channels)} channel(s), {n_images} image(s)."
        )

        self.scan_progress.setRange(0, 1)
        self.scan_progress.setValue(1)
        self.scan_progress.setFormat("Scan complete (%p%)")

    def _scan_root(self, root: Path) -> Generator[tuple[int, int, str], None, DatasetIndex]:
        max_workers = self._scan_worker_count()
        fov_dirs = []
        for child in root.iterdir():
            if child.is_dir():
                match = re.fullmatch(r"fov_(\d+)", child.name, re.IGNORECASE)
                if match:
                    fov_dirs.append((int(match.group(1)), child))

        if not fov_dirs:
            raise ValueError("No fov_<number> folders found.")

        index: dict[int, dict[str, dict[int, Path]]] = {}
        malformed: list[Path] = []

        scan_jobs: list[tuple[int, Path]] = []
        yield 0, len(fov_dirs), "Finding channel folders"
        for completed, (fov_number, fov_dir) in enumerate(sorted(fov_dirs), start=1):
            for channel_dir in sorted((p for p in fov_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                scan_jobs.append((fov_number, channel_dir))
            yield completed, len(fov_dirs), "Finding channel folders"

        if not scan_jobs:
            raise ValueError("No channel folders found.")

        yield 0, len(scan_jobs), "Scanning folders"
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._scan_channel_dir, fov_number, channel_dir)
                for fov_number, channel_dir in scan_jobs
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                fov_number, channel_index, channel_malformed = future.result()
                malformed.extend(channel_malformed)
                for channel, times in channel_index.items():
                    index.setdefault(fov_number, {}).setdefault(channel, {}).update(times)
                yield completed, len(scan_jobs), "Scanning folders"

        if malformed:
            print("Malformed TIFF filenames skipped:")
            for path in malformed[:50]:
                print(f"  {path}")
            if len(malformed) > 50:
                print(f"  ... {len(malformed) - 50} more")

        if not index:
            raise ValueError("No TIFF files matched the expected filename pattern.")

        fovs = sorted(index)
        channels = sorted({channel for fov_data in index.values() for channel in fov_data})
        if not channels:
            raise ValueError("No channels found.")

        sample_shape_by_channel: dict[str, tuple[int, ...]] = {}
        sample_dtype_by_channel: dict[str, np.dtype] = {}
        metadata_jobs = [
            (channel, sample_path)
            for channel in channels
            if (sample_path := self._first_path_for_channel(index, channel)) is not None
        ]
        yield 0, len(metadata_jobs), "Reading TIFF metadata"
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(metadata_jobs)))) as executor:
            futures = [
                executor.submit(self._read_sample_metadata, channel, sample_path)
                for channel, sample_path in metadata_jobs
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                channel, shape, dtype = future.result()
                if shape is not None and dtype is not None:
                    sample_shape_by_channel[channel] = shape
                    sample_dtype_by_channel[channel] = dtype
                yield completed, len(metadata_jobs), "Reading TIFF metadata"

        return DatasetIndex(
            root=root,
            index=index,
            fovs=fovs,
            channels=channels,
            sample_shape_by_channel=sample_shape_by_channel,
            sample_dtype_by_channel=sample_dtype_by_channel,
        )

    def _scan_worker_count(self) -> int:
        return max(4, min(32, (cpu_count() or 4) * 2))

    def _scan_channel_dir(
        self, fov_number: int, channel_dir: Path
    ) -> tuple[int, dict[str, dict[int, Path]], list[Path]]:
        channel_index: dict[str, dict[int, Path]] = {}
        malformed: list[Path] = []
        try:
            paths = sorted(channel_dir.iterdir(), key=lambda p: p.name)
        except Exception:
            traceback.print_exc()
            print(f"Could not list channel folder: {channel_dir}")
            return fov_number, channel_index, malformed

        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
                continue
            parsed = self._parse_tiff_name(path.name)
            if parsed is None:
                malformed.append(path)
                continue
            filename_fov, channel, time_number = parsed
            if filename_fov != fov_number:
                print(f"Skipping {path}: filename FOV does not match parent folder.")
                continue
            channel_index.setdefault(channel, {})[time_number] = path
        return fov_number, channel_index, malformed

    def _read_sample_metadata(
        self, channel: str, sample_path: Path
    ) -> tuple[str, tuple[int, ...] | None, np.dtype | None]:
        try:
            with tifffile.TiffFile(sample_path) as tif:
                series = tif.series[0]
                return channel, tuple(series.shape), np.dtype(series.dtype)
        except Exception:
            traceback.print_exc()
            print(f"Could not read sample metadata for channel {channel}: {sample_path}")
            return channel, None, None

    def _parse_tiff_name(self, filename: str) -> tuple[int, str, int] | None:
        match = TIFF_PATTERN.match(filename)
        if not match:
            return None
        fov = int(match.group(1))
        channel = match.group(2).strip("_")
        time_number = int(match.group(3))
        if not channel:
            return None
        return fov, channel, time_number

    def _first_path_for_channel(
        self, index: dict[int, dict[str, dict[int, Path]]], channel: str
    ) -> Path | None:
        for fov in sorted(index):
            times = index[fov].get(channel, {})
            if times:
                first_time = min(times)
                return times[first_time]
        return None

    def _populate_dataset_controls(self) -> None:
        if self.dataset is None:
            return
        self._updating_ui = True
        try:
            self.fov_combo.clear()
            for fov in self.dataset.fovs:
                self.fov_combo.addItem(f"fov_{fov}", fov)

            self.reference_combo.clear()
            reference_index = 0
            for i, channel in enumerate(self.dataset.channels):
                self.reference_combo.addItem(channel, channel)
                if channel == "bf":
                    reference_index = i
            self.reference_combo.setCurrentIndex(reference_index)
        finally:
            self._updating_ui = False

    def rebuild_channel_controls(self) -> None:
        self._clear_layout(self.channels_layout)
        self.channel_controls.clear()
        if self.dataset is None:
            return

        for channel in self.dataset.channels:
            channel_box = QGroupBox(channel)
            layout = QGridLayout(channel_box)

            visible_check = QCheckBox("Visible")
            visible_check.setChecked(True)

            min_spin = QSpinBox()
            max_spin = QSpinBox()
            for spin in (min_spin, max_spin):
                spin.setRange(-2147483648, 2147483647)

            dtype = self.dataset.sample_dtype_by_channel.get(channel)
            default_min, default_max = self._default_contrast_limits(dtype)
            min_spin.setValue(default_min)
            max_spin.setValue(default_max)

            opacity_slider = QSlider(Qt.Horizontal)
            opacity_slider.setRange(0, 100)
            opacity_slider.setValue(100 if channel == "bf" else 70)

            auto_button = QPushButton("Auto contrast current frame")

            layout.addWidget(visible_check, 0, 0, 1, 2)
            layout.addWidget(QLabel("Min"), 1, 0)
            layout.addWidget(min_spin, 1, 1)
            layout.addWidget(QLabel("Max"), 2, 0)
            layout.addWidget(max_spin, 2, 1)
            layout.addWidget(QLabel("Opacity"), 3, 0)
            layout.addWidget(opacity_slider, 3, 1)
            layout.addWidget(auto_button, 4, 0, 1, 2)

            self.channels_layout.addWidget(channel_box)
            self.channel_controls[channel] = {
                "visible": visible_check,
                "min": min_spin,
                "max": max_spin,
                "opacity": opacity_slider,
                "auto": auto_button,
            }

            visible_check.toggled.connect(self._channel_visibility_changed)
            min_spin.valueChanged.connect(lambda _value, c=channel: self._apply_contrast(c))
            max_spin.valueChanged.connect(lambda _value, c=channel: self._apply_contrast(c))
            opacity_slider.valueChanged.connect(lambda _value, c=channel: self._apply_opacity(c))
            auto_button.clicked.connect(lambda _checked=False, c=channel: self.auto_contrast(c))

        self.channels_layout.addStretch(1)

    def _default_contrast_limits(self, dtype: np.dtype | None) -> tuple[int, int]:
        if dtype is None:
            return 0, 65535
        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return int(max(info.min, 0)), int(info.max)
        if np.issubdtype(dtype, np.floating):
            return 0, 1
        return 0, 65535

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _timeline_control_changed(self) -> None:
        if self._updating_ui:
            return
        self.recompute_timeline()

    def _channel_visibility_changed(self) -> None:
        if self.dataset is None:
            return
        self.recompute_timeline()

    def recompute_timeline(self) -> None:
        if self.dataset is None:
            self.timeline = []
            self._update_time_controls()
            self.stop_playback()
            return

        previous_real_t = self.current_real_t
        fov = self.current_fov()
        if fov is None:
            self.timeline = []
        elif self.mode_combo.currentText() == self.COMPLETE_CHANNELS:
            self.timeline = self._complete_channels_timeline(fov)
        else:
            self.timeline = self._full_timeline(fov)

        if self.timeline:
            if previous_real_t is None:
                frame_index = 0
            else:
                frame_index = min(
                    range(len(self.timeline)),
                    key=lambda i: abs(self.timeline[i] - previous_real_t),
                )
            self.current_real_t = self.timeline[frame_index]
        else:
            frame_index = 0
            self.current_real_t = None

        self._update_time_controls(frame_index)
        if not self.timeline:
            self.stop_playback()
        self.update_display()

    def _full_timeline(self, fov: int) -> list[int]:
        reference = self.current_reference_channel()
        if reference is None:
            self._set_status("No reference channel selected.")
            return []
        times = self.dataset.index.get(fov, {}).get(reference, {}) if self.dataset else {}
        if not times:
            self._set_status(f"Selected FOV has no images for reference channel {reference}.")
            return []
        return sorted(times)

    def _complete_channels_timeline(self, fov: int) -> list[int]:
        if self.dataset is None:
            return []
        channels = self.selected_channels()
        if not channels:
            self._set_status("No visible channels selected.")
            return []
        time_sets = []
        for channel in channels:
            times = set(self.dataset.index.get(fov, {}).get(channel, {}))
            if not times:
                self._set_status(f"Selected FOV/channel has no images: fov_{fov}, {channel}.")
                return []
            time_sets.append(times)
        return sorted(set.intersection(*time_sets)) if time_sets else []

    def _update_time_controls(self, frame_index: int = 0) -> None:
        old_start = self.play_start_spin.value()
        old_end = self.play_end_spin.value()
        self._updating_ui = True
        try:
            maximum = max(0, len(self.timeline) - 1)
            self.time_slider.setRange(0, maximum)
            self.time_spin.setRange(0, maximum)
            self.play_start_spin.setRange(0, maximum)
            self.play_end_spin.setRange(0, maximum)
            frame_index = min(max(0, frame_index), maximum)
            self.time_slider.setValue(frame_index)
            self.time_spin.setValue(frame_index)
            self.play_start_spin.setValue(min(old_start, maximum))
            self.play_end_spin.setValue(max(min(old_end, maximum), min(old_start, maximum)))
            if len(self.timeline) > 0 and old_start == 0 and old_end == 0:
                self.play_end_spin.setValue(maximum)
        finally:
            self._updating_ui = False

    def _slider_changed(self, value: int) -> None:
        if self._updating_ui:
            return
        self._set_frame_index(value)

    def _spin_changed(self, value: int) -> None:
        if self._updating_ui:
            return
        self._set_frame_index(value)

    def _set_frame_index(self, frame_index: int) -> None:
        if not self.timeline:
            return
        frame_index = min(max(0, frame_index), len(self.timeline) - 1)
        self.current_real_t = self.timeline[frame_index]
        self._updating_ui = True
        try:
            self.time_slider.setValue(frame_index)
            self.time_spin.setValue(frame_index)
        finally:
            self._updating_ui = False
        self.update_display()

    def toggle_playback(self) -> None:
        if self.play_timer.isActive():
            self.stop_playback()
            return
        self.start_playback()

    def start_playback(self) -> None:
        if not self.timeline:
            self._set_status("No frames available for playback.")
            return
        start_index, end_index = self.playback_range()
        current_index = self.current_frame_index()
        if current_index < start_index or current_index > end_index:
            self._set_frame_index(start_index)
        self.play_button.setText("Pause")
        self.play_timer.start(self.play_interval_ms())

    def stop_playback(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
        if hasattr(self, "play_button"):
            self.play_button.setText("Play")

    def playback_range(self) -> tuple[int, int]:
        if not self.timeline:
            return 0, 0
        maximum = len(self.timeline) - 1
        start_index = min(max(0, self.play_start_spin.value()), maximum)
        end_index = min(max(0, self.play_end_spin.value()), maximum)
        if end_index < start_index:
            start_index, end_index = end_index, start_index
        return start_index, end_index

    def play_interval_ms(self) -> int:
        fps = max(0.1, self.play_fps_spin.value())
        return max(1, int(round(1000.0 / fps)))

    def _advance_playback(self) -> None:
        if not self.timeline:
            self.stop_playback()
            return
        start_index, end_index = self.playback_range()
        current_index = self.current_frame_index()
        next_index = start_index if current_index >= end_index else current_index + 1
        if next_index < start_index:
            next_index = start_index
        self._set_frame_index(next_index)

    def _play_range_changed(self) -> None:
        if self._updating_ui:
            return
        if not self.timeline:
            return
        start_index, end_index = self.playback_range()
        if self.current_frame_index() < start_index or self.current_frame_index() > end_index:
            self._set_frame_index(start_index)

    def _play_speed_changed(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.start(self.play_interval_ms())

    def update_display(self) -> None:
        if self.dataset is None:
            self._update_time_label()
            return
        if self.current_real_t is None or not self.timeline:
            self._update_time_label()
            self._set_status("No frames available for the selected settings.")
            return

        fov = self.current_fov()
        if fov is None:
            return

        real_t = self.current_real_t
        status_lines = [
            f"Frame index {self.current_frame_index()} / {max(0, len(self.timeline) - 1)}",
            f"Real t = {real_t}",
        ]

        for channel in self.dataset.channels:
            controls = self.channel_controls.get(channel)
            channel_visible = bool(controls and controls["visible"].isChecked())
            if channel_visible:
                image, message = self.load_channel_image(channel, real_t)
                self._ensure_layer(channel, image)
                self._set_layer_data_preserving_contrast(channel, image)
                self.layers[channel].visible = True
                self._apply_opacity(channel)
                status_lines.append(message)
            elif channel in self.layers:
                self.layers[channel].visible = False

        self._update_time_label()
        self._set_status("\n".join(status_lines))

    def load_channel_image(self, channel: str, real_t: int) -> tuple[np.ndarray, str]:
        fov = self.current_fov()
        if self.dataset is None or fov is None:
            return np.zeros((1, 1), dtype=np.uint8), "No dataset loaded."

        path = self.dataset.index.get(fov, {}).get(channel, {}).get(real_t)
        if path is None:
            return (
                self.make_blank(channel, fov),
                f"{channel}: missing at t={real_t}, blank displayed",
            )

        key = (fov, channel, real_t)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, f"{channel}: loaded t={real_t} (cache)"

        try:
            image = tifffile.imread(path)
        except Exception as exc:
            traceback.print_exc()
            return (
                self.make_blank(channel, fov),
                f"{channel}: failed to read t={real_t}, blank displayed ({exc})",
            )

        self.dataset.sample_shape_by_channel.setdefault(channel, tuple(image.shape))
        self.dataset.sample_dtype_by_channel.setdefault(channel, image.dtype)
        self.cache.put(key, image)
        return image, f"{channel}: loaded t={real_t}"

    def make_blank(self, channel: str, fov: int) -> np.ndarray:
        if self.dataset is None:
            return np.zeros((1, 1), dtype=np.uint8)
        shape = self.dataset.sample_shape_by_channel.get(channel)
        dtype = self.dataset.sample_dtype_by_channel.get(channel)

        if shape is None or dtype is None:
            for layer in self.layers.values():
                shape = tuple(layer.data.shape)
                dtype = layer.data.dtype
                break

        if (shape is None or dtype is None) and self.current_reference_channel() is not None:
            reference = self.current_reference_channel()
            shape = self.dataset.sample_shape_by_channel.get(reference)
            dtype = self.dataset.sample_dtype_by_channel.get(reference)

        if shape is None or dtype is None:
            shape = (1, 1)
            dtype = np.uint8

        return np.zeros(shape, dtype=dtype)

    def _ensure_layer(self, channel: str, image: np.ndarray) -> None:
        if channel in self.layers:
            return
        colormap = self._colormap_for_channel(channel)
        opacity = 1.0 if channel == "bf" else 0.7
        layer = self.viewer.add_image(
            image,
            name=channel,
            colormap=colormap,
            blending="additive",
            opacity=opacity,
        )
        self.layers[channel] = layer
        self._apply_contrast(channel)
        self._apply_opacity(channel)

    def _set_layer_data_preserving_contrast(self, channel: str, image: np.ndarray) -> None:
        layer = self.layers[channel]
        previous_limits = tuple(layer.contrast_limits)
        layer.data = image
        layer.contrast_limits = previous_limits

    def _colormap_for_channel(self, channel: str) -> str:
        normalized = channel.lower()
        if normalized in {"bf", "brightfield", "gray", "grey"}:
            return "gray"
        if normalized == "yellow":
            return "yellow"
        if normalized in {"green", "gfp"}:
            return "green"
        if normalized in {"red", "rfp"}:
            return "red"
        if normalized in {"blue", "dapi"}:
            return "blue"
        return "gray"

    def _apply_contrast(self, channel: str) -> None:
        layer = self.layers.get(channel)
        controls = self.channel_controls.get(channel)
        if layer is None or controls is None:
            return
        min_value = controls["min"].value()
        max_value = controls["max"].value()
        if max_value <= min_value:
            max_value = min_value + 1
        layer.contrast_limits = (min_value, max_value)

    def _apply_opacity(self, channel: str) -> None:
        layer = self.layers.get(channel)
        controls = self.channel_controls.get(channel)
        if layer is None or controls is None:
            return
        layer.opacity = controls["opacity"].value() / 100.0

    def auto_contrast(self, channel: str) -> None:
        layer = self.layers.get(channel)
        controls = self.channel_controls.get(channel)
        if layer is None or controls is None:
            self._set_status(f"No displayed image for channel {channel}.")
            return

        image = np.asarray(layer.data)
        if image.size == 0 or np.all(image == 0):
            self._set_status(f"{channel}: auto contrast ignored all-zero blank image.")
            return

        low, high = np.percentile(image, [0.5, 99.5])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            self._set_status(f"{channel}: could not compute useful contrast limits.")
            return

        self._updating_ui = True
        try:
            controls["min"].setValue(int(np.floor(low)))
            controls["max"].setValue(int(np.ceil(high)))
        finally:
            self._updating_ui = False
        self._apply_contrast(channel)
        self._set_status(f"{channel}: auto contrast set to {low:.3g} - {high:.3g}.")

    def selected_channels(self) -> list[str]:
        channels = []
        for channel, controls in self.channel_controls.items():
            if controls["visible"].isChecked():
                channels.append(channel)
        return channels

    def current_fov(self) -> int | None:
        if self.fov_combo.count() == 0:
            return None
        return self.fov_combo.currentData()

    def current_reference_channel(self) -> str | None:
        if self.reference_combo.count() == 0:
            return None
        return self.reference_combo.currentData()

    def current_frame_index(self) -> int:
        if not self.timeline or self.current_real_t is None:
            return 0
        try:
            return self.timeline.index(self.current_real_t)
        except ValueError:
            return self.time_spin.value()

    def _update_time_label(self) -> None:
        total = len(self.timeline)
        if total == 0 or self.current_real_t is None:
            self.time_label.setText("Frame index: 0 / 0\nReal timepoint: t = -")
            return
        self.time_label.setText(
            f"Frame index: {self.current_frame_index()} / {total - 1}\n"
            f"Real timepoint: t = {self.current_real_t}"
        )

    def next_frame(self, viewer: napari.Viewer | None = None) -> None:
        if not self.timeline:
            return
        self._set_frame_index(min(self.current_frame_index() + 1, len(self.timeline) - 1))

    def prev_frame(self, viewer: napari.Viewer | None = None) -> None:
        if not self.timeline:
            return
        self._set_frame_index(max(self.current_frame_index() - 1, 0))

    def next_fov(self, viewer: napari.Viewer | None = None) -> None:
        if self.fov_combo.count() == 0:
            return
        self.fov_combo.setCurrentIndex(min(self.fov_combo.currentIndex() + 1, self.fov_combo.count() - 1))

    def prev_fov(self, viewer: napari.Viewer | None = None) -> None:
        if self.fov_combo.count() == 0:
            return
        self.fov_combo.setCurrentIndex(max(self.fov_combo.currentIndex() - 1, 0))

    def _clear_cache(self) -> None:
        self.cache.clear()
        self._set_status("Cache cleared.")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moma-view-timelapse",
        description="Lazy napari timelapse viewer over a dataset root of "
        "fov_<n>/<channel>/fov_<n>_<channel>_t<k>.tif(f) trees.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Dataset root folder to open and scan at startup.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    viewer = napari.Viewer()
    widget = LazyTimelapseWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right", name="Lazy Timelapse Viewer")
    if args.root:
        widget.root_edit.setText(args.root)
        widget.scan_dataset()
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
