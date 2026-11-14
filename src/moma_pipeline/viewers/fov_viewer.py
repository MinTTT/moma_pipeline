"""napari FOV export viewer.

Ported from Moma_cellPose/Moma_CPM_FOV_viewer.py into
moma_pipeline.viewers.fov_viewer (src layout, absolute imports).

Refactors relative to the old script:

- All previously-module-global state lives on :class:'FovViewer'; the latent
  UnboundLocalError risk of the old 'change_time_slice' is gone because
  nothing is shared through globals anymore.
- Segmentation goes through 'moma_pipeline.models.SingleModelSegmenter'
  only; cellpose is never imported here. The segmenter is built lazily when
  "Run Model" is pressed, from the checkpoint selected in the dropdown.
- The model dropdown lists explicit checkpoints discovered under
  'moma_pipeline.models.DEFAULT_MODEL_ROOT' (one level of subfolders, like
  the old os.listdir('./CPModels')) plus any --model_dir directories.
- All optical channels have synchronized image layers; missing channel frames
  display zeros without being inserted into the source export dictionaries.
- No hardcoded machine paths: the exported-data directory comes from the
  positional CLI argument (default: current working directory).
- Bare 'except:' swallows were replaced by 'except Exception' that logs
  the actual error message (the error dialog behaviour is kept).

Importing this module creates no QApplication and imports no cellpose; both
happen only when FovViewer is instantiated / the model runs.

Run:

    python -m moma_pipeline.viewers.fov_viewer [exported_data_dir] [--model_dir DIR ...]
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import napari
from magicgui import magicgui, widgets
from skimage import io

from moma_pipeline.core.fov_utils import get_time_now
from moma_pipeline.io.fov_pkl import load_Moma_fov_data_dt
from moma_pipeline.models import (
    DEFAULT_CELL_MODEL_PATH,
    DEFAULT_MODEL_ROOT,
    SingleModelSegmenter,
)

__all__ = [
    "FovViewer",
    "build_parser",
    "main",
    "load_Moma_fov_data",
    "show_error_message",
]

# Placeholder dropdown entry for empty choice lists (magicgui Dropdown rejects
# value=None when there are no choices; read callbacks treat it as 'nothing').
_EMPTY_CHOICE = "(none)"


def load_Moma_fov_data(data_dict):
    """Parse an exported DT pickle dict into the legacy 10-tuple.

    Same contract as the old Moma_CPM_FOV_viewer.load_Moma_fov_data
    (info_dict + cells_sideChannel_image_dict keys), delegated to the
    package loader moma_pipeline.io.fov_pkl.load_Moma_fov_data_dt so
    this viewer opens DT pickles produced by this package. Returns:

        (fov_cells_dict, fov_channels_image_dict, fov_cells_mask_dict,
         time_points_seconds, time_points, opticalChannels, other_channels,
         cell_detection_opticalChannel_name, channel_keys,
         images_background_fluorescence)
    """
    return load_Moma_fov_data_dt(data_dict)


def show_error_message(message: str):
    """Pop up a modal magicgui dialog showing 'message' (as in the old viewer)."""

    @magicgui(call_button="OK", layout="vertical")
    def error_dialog():
        return message

    error_dialog.show_modal()


def _collect_model_entries(
    extra_model_dirs: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Optional[str]]]:
    """Collect (display_name, absolute_checkpoint_path) model entries.

    Sources: DEFAULT_MODEL_ROOT plus every directory in
    'extra_model_dirs'. Each root is scanned one level deep - root entries are
    checkpoints, subfolder entries become 'subfolder/checkpoint' - matching
    the old os.listdir('./CPModels') listing but yielding explicit paths.
    A final ('None', None) entry mirrors the old 'None' choice.
    """
    entries: List[Tuple[str, Optional[str]]] = []
    seen_paths = set()
    seen_names = set()
    roots: List[str] = [DEFAULT_MODEL_ROOT]
    roots.extend(str(d) for d in (extra_model_dirs or []) if d)
    for root in roots:
        if not root or not os.path.isdir(root):
            print(f"{get_time_now()} FOV Viewer -> model directory not found, skipped: {root}")
            continue
        root = os.path.abspath(root)
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                children = [os.path.join(path, sub) for sub in sorted(os.listdir(path))]
                candidates = [(f"{name}/{os.path.basename(sub)}", sub) for sub in children]
            else:
                candidates = [(name, path)]
            for display, full_path in candidates:
                key = os.path.normcase(os.path.abspath(full_path))
                if key in seen_paths or display in seen_names:
                    continue
                seen_paths.add(key)
                seen_names.add(display)
                entries.append((display, full_path))
    entries.append(("None", None))
    return entries


def _default_model_display(entries: Sequence[Tuple[str, Optional[str]]]) -> str:
    """Display name of the package cell model checkpoint, when listed."""
    target = os.path.normcase(os.path.abspath(DEFAULT_CELL_MODEL_PATH))
    for display, path in entries:
        if path and os.path.normcase(os.path.abspath(path)) == target:
            return display
    base = os.path.basename(DEFAULT_CELL_MODEL_PATH.rstrip(os.sep))
    for display, path in entries:
        if display == base or (path and os.path.basename(path.rstrip(os.sep)) == base):
            return display
    return "None"


class FovViewer:
    """MOMA FOV export viewer; owns all state the old script kept in globals."""

    IMAGE_LAYER_NAME = "cells_image"
    MASK_LAYER_NAME = "cell_mask"

    def __init__(self, exported_data_dir: Optional[str] = None,
                 model_dirs: Optional[Sequence[str]] = None):
        self.exported_data_dir = str(exported_data_dir) if exported_data_dir else os.getcwd()
        self.extra_model_dirs: List[str] = [str(d) for d in (model_dirs or [])]

        # --- exported FOV data state (old module globals) ---
        self.exported_data_name_list: List[str] = []
        self.exported_data_path_dict: Dict[str, str] = {}
        self.fov_name: str = ""
        self.fov_cells_dict: Dict = {}
        self.fov_channels_image_dict: Dict = {}
        self.fov_cells_mask_dict: Dict = {}
        self.time_points_seconds = None
        self.time_points: Sequence = []
        self.opticalChannels: List = []
        self.other_channels: List = []
        self.cell_detection_opticalChannel_name: str = ""
        self.channel_keys: List = []
        self.fluorescence_background = None
        self.time_points_slice = slice(0, 0)
        self.time_points_part: Sequence = []
        self.channel_index = None
        self.cells_image_stack: Optional[np.ndarray] = None
        self.cells_mask_stack: Optional[np.ndarray] = None
        self.napri_cell_image = None
        self.napri_cell_mask = None
        # One synchronized image layer per optical channel; legacy aliases above
        # still refer to the detection channel for segmentation and mask saving.
        self.optical_image_stacks: Dict[str, np.ndarray] = {}
        self.optical_image_layers: Dict[str, object] = {}
        self.optical_missing_frames: Dict[str, np.ndarray] = {}
        self._optical_contrast_initialized = set()

        # --- segmentation state (old CP_MODEL global) ---
        self.model_entries = _collect_model_entries(self.extra_model_dirs)
        self.model_paths: Dict[str, Optional[str]] = dict(self.model_entries)
        self.model_segmenter: Optional[SingleModelSegmenter] = None
        self.model_checkpoint: Optional[str] = None

        # --- napari viewer + docks ---
        self.viewer = napari.Viewer()
        self._build_fov_selector_dock()
        self._build_segmentation_dock()
        self._build_side_channel_dock()
        self.viewer.window._qt_window.setWindowTitle("MOMA FOV Viewer")
        self._scan_exported_data_dir()
        if self.exported_data_name_list:
            self.fov_file_selector.choices = self.exported_data_name_list
            self.fov_file_selector.value = self.exported_data_name_list[0]
            self.read_fov_file()
        else:
            self._set_time_range_controls([])

    # ------------------------------------------------------------------ docks
    def _build_fov_selector_dock(self) -> None:
        self.fov_dir_selector = widgets.FileEdit(
            name="FOV_dir", value=self.exported_data_dir, mode="d")
        fov_choices = self.exported_data_name_list or [_EMPTY_CHOICE]
        self.fov_file_selector = widgets.Dropdown(
            name="FOV_file",
            choices=lambda widget: self.exported_data_name_list or [_EMPTY_CHOICE],
            value=fov_choices[0])
        self.fov_file_read_button = widgets.PushButton(name="Read Fov", text="Read Fov")
        self.fov_file_current_box = widgets.Label(name="Current FOV", value=self.fov_name)

        self.fov_dir_selector.changed.connect(self.read_fov_dir)
        self.fov_file_read_button.changed.connect(self.read_fov_file)

        self.fov_selector_module = widgets.Container(
            widgets=[self.fov_dir_selector, self.fov_file_selector,
                     self.fov_file_read_button, self.fov_file_current_box],
            layout="vertical")
        self.viewer.window.add_dock_widget(
            self.fov_selector_module, area="right", name="FOV_selector")

    def _build_segmentation_dock(self) -> None:
        model_names = [display for display, _ in self.model_entries]
        self.model_selector = widgets.Dropdown(
            name="Model", choices=model_names, value=_default_model_display(self.model_entries))
        self.model_load_button = widgets.PushButton(name="Load Model", text="Load Model")
        self.model_load_button.changed.connect(self.load_model)
        self.model_pars_diameter = widgets.FloatText(name="Diameter", value=30, step=1)
        self.model_pars_cellprob = widgets.FloatSlider(name="CellProb", min=-6, max=6, value=0)
        self.model_pars_flow = widgets.FloatText(name="Flow", value=0.4, step=0.1)
        # number of mask recomputation iterations, Cellpose auto default at
        # diameter 30 is 200, increase to better segment long cells
        self.model_pars_niter = widgets.IntText(name="niter", value=200, min=1, max=2000, step=10)
        self.model_run_button = widgets.PushButton(name="Run Model", text="Run Model")
        self.model_run_button.changed.connect(self.evaluate_image)
        self.model_segmentation_container = widgets.Container(
            widgets=[self.model_selector, self.model_load_button,
                     widgets.Container(widgets=[self.model_pars_diameter,
                                                self.model_pars_cellprob,
                                                self.model_pars_flow,
                                                self.model_pars_niter]),
                     self.model_run_button],
            layout="vertical")
        self.viewer.window.add_dock_widget(
            self.model_segmentation_container, area="right", name="Segmentation")

    def _build_side_channel_dock(self) -> None:
        self.slice_slider = widgets.RangeSlider(
            name="time_slice", min=0, max=0, value=[0, 0],
            tooltip="Inclusive time range: (0, 0) displays the first frame")
        self.slice_slider.changed.connect(self.change_time_slice)

        channel_choices = self.channel_keys or [_EMPTY_CHOICE]
        self.channel_selection = widgets.Dropdown(
            name="channel",
            choices=lambda widget: self.channel_keys or [_EMPTY_CHOICE],
            value=channel_choices[0])
        self.channel_selection.changed.connect(self.change_channel)

        self.save_button = widgets.PushButton(name="Save", text="Save")
        self.dir_selector = widgets.FileEdit(name="Save_dir", value="./cells_mask", mode="d")
        self.save_module = widgets.Container(
            widgets=[self.dir_selector, self.save_button], layout="vertical")
        self.save_button.changed.connect(self.save_mask)

        self.viewer.window.add_dock_widget(
            widgets.Container(widgets=[self.slice_slider, self.channel_selection,
                                       self.save_module], layout="vertical"),
            area="right", name="Side channel selector")

    # ------------------------------------------------------------- FOV I/O
    def _scan_exported_data_dir(self) -> None:
        """(Re)scan exported_data_dir for *exported_data.pkl files."""
        self.read_fov_dir(update_widget=False)

    def read_fov_dir(self, x=None, update_widget: bool = True) -> None:
        """List exported-data pickles in the selected directory (old read_fov_dir)."""
        fov_dir = str(self.fov_dir_selector.value)
        if update_widget:
            self.exported_data_dir = fov_dir
        try:
            names = [name for name in os.listdir(fov_dir) if "exported_data.pkl" in name]
        except OSError as exc:
            print(f"{get_time_now()} FOV Viewer -> cannot list {fov_dir}: {exc}")
            return
        # try to sort name
        try:
            names.sort(key=lambda name: int(name.split("_")[1]))
        except (ValueError, IndexError):
            names.sort()
        self.exported_data_name_list = names
        self.exported_data_path_dict = {
            name: os.path.join(fov_dir, name) for name in names}
        if update_widget:
            self.fov_file_selector.choices = names or [_EMPTY_CHOICE]
            self.fov_file_selector.value = names[0] if names else _EMPTY_CHOICE
        else:
            self.fov_file_selector.choices = names or [_EMPTY_CHOICE]

    def read_fov_file(self, x=None) -> None:
        """Load the selected exported-data pickle into the viewer (old read_fov_file)."""
        fov_file_name = self.fov_file_selector.value
        if fov_file_name not in self.exported_data_path_dict:
            print(f"{get_time_now()} FOV Viewer -> no FOV file selected "
                  f"({fov_file_name!r}); nothing to read")
            return
        self.fov_name = "_".join(str(fov_file_name).split("_")[:2])
        fov_path = self.exported_data_path_dict[fov_file_name]
        with open(fov_path, "rb") as file:
            exported_data = pickle.load(file)
        (self.fov_cells_dict, self.fov_channels_image_dict, self.fov_cells_mask_dict,
         self.time_points_seconds, self.time_points, self.opticalChannels,
         self.other_channels, self.cell_detection_opticalChannel_name,
         self.channel_keys, self.fluorescence_background) = load_Moma_fov_data(exported_data)

        # Read every frame. The old slice(0, -1) silently dropped the last one.
        # Configure ranges before any channel-selection callback can render.
        self._set_time_range_controls(self.time_points)
        self.channel_keys = sorted(self.channel_keys, key=lambda val: int(val))
        with self.channel_selection.changed.blocked():
            self.channel_selection.choices = self.channel_keys or [_EMPTY_CHOICE]
            self.channel_selection.value = (self.channel_keys[0] if self.channel_keys
                                            else _EMPTY_CHOICE)
        self.fov_file_current_box.value = self.fov_name
        if self.channel_keys and self.time_points:
            self.change_channel(self.channel_keys[0])
        else:
            self._clear_data_layers()
            print(f"{get_time_now()} FOV Viewer -> {self.fov_name} has no "
                  "channel/time-point data; nothing to display")

    # -------------------------------------------------------- layer updates
    def _set_time_range_controls(self, time_points: Sequence) -> None:
        """Set slider bounds and a full, inclusive range without emitting callbacks."""
        count = len(time_points)
        last = max(0, count - 1)
        with self.slice_slider.changed.blocked():
            self.slice_slider.min = 0
            self.slice_slider.max = last
            self.slice_slider.value = (0, last)
        self.slice_slider.enabled = bool(count and self.channel_keys)
        self.time_points_slice = slice(0, count)
        self.time_points_part = self.time_points[self.time_points_slice]

    def _clear_data_layers(self) -> None:
        """Clear displayed and cached data when the loaded export is empty."""
        for layer in [*self.optical_image_layers.values(), self.napri_cell_mask]:
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self.channel_index = None
        self.cells_image_stack = None
        self.cells_mask_stack = None
        self.napri_cell_image = None
        self.napri_cell_mask = None
        self.optical_image_stacks.clear()
        self.optical_image_layers.clear()
        self.optical_missing_frames.clear()
        self._optical_contrast_initialized.clear()

    def _optical_channel_names(self) -> List[str]:
        """Keep declared channels, including all-missing ones, and discovered ones."""
        names = [self.cell_detection_opticalChannel_name, *self.opticalChannels]
        for side_images in self.fov_channels_image_dict.values():
            for frame_images in side_images.values():
                if frame_images is not None:
                    names.extend(frame_images)
        return list(dict.fromkeys(name for name in names if name))

    def _build_optical_stacks(self, side: int) -> Tuple[Dict[str, np.ndarray],
                                                       Dict[str, np.ndarray],
                                                       np.ndarray]:
        """Align every channel to the shared timeline without altering source data."""
        side_images = self.fov_channels_image_dict.get(side, {})
        side_masks = self.fov_cells_mask_dict.get(side, {})
        channels = self._optical_channel_names()
        detection = self.cell_detection_opticalChannel_name
        reference = None
        # Cropped images and masks share their spatial shape. Prefer phase/BF,
        # but allow a fluorescence image or mask to supply the blank-frame shape.
        for time in self.time_points:
            frame = side_images.get(time) or {}
            candidates = [frame.get(detection), *frame.values(), side_masks.get(time)]
            reference = next((np.asarray(image) for image in candidates
                              if image is not None and np.asarray(image).ndim == 2), None)
            if reference is not None:
                break
        shape = reference.shape if reference is not None else (1, 1)
        default_dtype = reference.dtype if reference is not None else np.dtype("uint16")
        count = len(self.time_points)
        stacks, missing_frames = {}, {}
        for channel in channels:
            frames = [(side_images.get(time) or {}).get(channel)
                      for time in self.time_points]
            available = [np.asarray(image) for image in frames if image is not None]
            for image in available:
                if image.shape != shape:
                    raise ValueError(f"Optical channel {channel!r} has shape "
                                     f"{image.shape}, expected {shape}")
            # Keep each channel's native dtype; blanks are display-only zeros.
            dtype = (np.result_type(*(image.dtype for image in available))
                     if available else default_dtype)
            stack = np.zeros((count, *shape), dtype=dtype)
            missing = np.ones(count, dtype=bool)
            for index, image in enumerate(frames):
                if image is not None:
                    stack[index] = image
                    missing[index] = False
            stacks[channel], missing_frames[channel] = stack, missing
        available_masks = [np.asarray(side_masks[time]) for time in self.time_points
                           if side_masks.get(time) is not None]
        mask_dtype = (np.result_type(*(mask.dtype for mask in available_masks))
                      if available_masks else np.dtype("uint32"))
        mask_stack = np.zeros((count, *shape), dtype=mask_dtype)
        for index, time in enumerate(self.time_points):
            mask = side_masks.get(time)
            if mask is not None:
                if np.asarray(mask).shape != shape:
                    raise ValueError(f"Mask at {time!r} does not match image shape {shape}")
                mask_stack[index] = mask
        return stacks, missing_frames, mask_stack

    def change_channel(self, x) -> None:
        """Switch the side channel while retaining all optical layers and time range."""
        if not self.time_points or x == _EMPTY_CHOICE:
            return
        try:
            channel_index = int(x)
            stacks, missing_frames, mask_stack = self._build_optical_stacks(channel_index)
        except (KeyError, TypeError, ValueError) as exc:
            self._clear_data_layers()
            print(f"{get_time_now()} FOV Viewer -> cannot load channel {x!r}: "
                  f"{type(exc).__name__}: {exc}")
            return
        print(f"{get_time_now()} SideChannel Viewer -> Change channel to {x}")
        self.channel_index = channel_index
        self.optical_image_stacks = stacks
        self.optical_missing_frames = missing_frames
        self.cells_image_stack = stacks[self.cell_detection_opticalChannel_name]
        self.cells_mask_stack = mask_stack
        self._apply_time_slice(self.time_points_slice.start,
                               self.time_points_slice.stop - 1)

    def _colormap_for_channel(self, channel: str) -> str:
        """Conventional display colors, customizable through napari layer controls."""
        normalized = channel.lower()
        if (channel == self.cell_detection_opticalChannel_name
                or normalized.startswith(("bf", "phase", "brightfield"))):
            return "gray"
        if any(token in normalized for token in ("green", "gfp")):
            return "green"
        if any(token in normalized for token in ("red", "rfp", "mcherry", "mscarlet")):
            return "red"
        if any(token in normalized for token in ("yellow", "yfp")):
            return "yellow"
        if any(token in normalized for token in ("blue", "dapi", "cfp")):
            return "blue"
        return "gray"

    def _initial_contrast_limits(self, channel: str) -> Tuple[float, float]:
        """Ignore zero placeholders when estimating initial contrast."""
        low, high = np.inf, -np.inf
        for image, missing in zip(self.optical_image_stacks[channel],
                                  self.optical_missing_frames[channel]):
            if not missing:
                finite = image[np.isfinite(image)]
                if finite.size:
                    low = min(low, float(finite.min()))
                    high = max(high, float(finite.max()))
        if not np.isfinite(low) or not np.isfinite(high):
            return (0.0, 1.0)
        return (low, high if high > low else low + 1.0)

    def _apply_time_slice(self, left: int, right: int) -> None:
        """Display inclusive [left, right]; equal endpoints show one frame."""
        if self.cells_image_stack is None:
            return
        if right < left:
            left, right = right, left
        last = len(self.cells_image_stack) - 1
        left = min(max(int(left), 0), last)
        right = min(max(int(right), 0), last)
        stop = right + 1
        self.time_points_slice = slice(left, stop)
        self.time_points_part = self.time_points[self.time_points_slice]
        # Remove obsolete optical layers when switching to a different FOV.
        for channel in list(self.optical_image_layers):
            if channel not in self.optical_image_stacks:
                layer = self.optical_image_layers.pop(channel)
                if layer in self.viewer.layers:
                    self.viewer.layers.remove(layer)
                self._optical_contrast_initialized.discard(channel)
        # Every optical layer uses the same timeline, crop and side-channel shape.
        for channel, stack in self.optical_image_stacks.items():
            data = stack[left:stop, ...]
            missing = self.optical_missing_frames[channel]
            layer = self.optical_image_layers.get(channel)
            if layer is None or layer not in self.viewer.layers:
                layer = self.viewer.add_image(
                    data, name=channel, rgb=False, blending="additive",
                    colormap=self._colormap_for_channel(channel),
                    contrast_limits=self._initial_contrast_limits(channel))
                self.optical_image_layers[channel] = layer
                if not missing.all():
                    self._optical_contrast_initialized.add(channel)
            else:
                # Retain the user's contrast, visibility, colormap and opacity.
                contrast_limits = tuple(layer.contrast_limits)
                layer.data = data
                layer.contrast_limits = contrast_limits
                if channel not in self._optical_contrast_initialized and not missing.all():
                    layer.contrast_limits = self._initial_contrast_limits(channel)
                    self._optical_contrast_initialized.add(channel)
            layer.metadata.update({
                "optical_channel": channel,
                "missing_time_points": [time for time, absent in
                                        zip(self.time_points_part, missing[left:stop]) if absent],
            })
        self.napri_cell_image = self.optical_image_layers[self.cell_detection_opticalChannel_name]
        if self.napri_cell_mask is None or self.napri_cell_mask not in self.viewer.layers:
            self.napri_cell_mask = self.viewer.add_labels(
                self.cells_mask_stack[left:stop, ...], name=self.MASK_LAYER_NAME,
                blending="additive", opacity=0.25)
        else:
            self.napri_cell_mask.data = self.cells_mask_stack[left:stop, ...]
        # Keep the cell-mask overlay above all image channels.
        self.viewer.layers.move(self.viewer.layers.index(self.napri_cell_mask),
                                len(self.viewer.layers) - 1)
        with self.slice_slider.changed.blocked():
            self.slice_slider.value = (left, right)
        self.viewer.dims.set_point(value=0, axis=0)

    def change_time_slice(self, x) -> None:
        """Select an inclusive time range; collapsed endpoints mean one frame."""
        try:
            left, right = x
        except (TypeError, ValueError):
            return
        self._apply_time_slice(left, right)

    def save_mask(self, x=None) -> None:
        """Save the current mask and image to the directory selected in the dock."""
        try:
            save_dir = str(self.dir_selector.value)
            # check path
            if not os.path.exists(save_dir):
                print(f"SideChannel Viewer -> Path {save_dir} does not exist, create it")
                os.makedirs(save_dir)
            # Layer axes are cropped, but filenames use the original time index.
            time_index = int(self.viewer.dims.point[0])
            source_index = self.time_points_slice.start + time_index
            if not 0 <= time_index < len(self.napri_cell_mask.data):
                raise IndexError(f"Displayed time index {time_index} is out of range")

            file_name = f"{self.fov_name}_channel_{self.channel_index}_t_{source_index}"
            time_now_fmt = get_time_now("2")
            mask_save_name = f"{time_now_fmt}_{file_name}_masks.tif"
            image_save_name = f"{time_now_fmt}_{file_name}.tif"
            mask_save_path = os.path.join(save_dir, mask_save_name)
            print(f"{get_time_now()} SideChannel Viewer -> Save mask to {mask_save_path}")
            image_save_path = os.path.join(save_dir, image_save_name)
            print(f"{get_time_now()} SideChannel Viewer -> Save image to {image_save_path}")

            io.imsave(mask_save_path, self.napri_cell_mask.data[time_index, ...],
                      check_contrast=False)
            io.imsave(image_save_path, self.napri_cell_image.data[time_index, ...],
                      check_contrast=False)
        except Exception as exc:
            # pop up a new window to show error
            show_error_message(f"An error occurred while saving the file: {exc}")
            print(f"{get_time_now()} SideChannel Viewer -> Error in saving mask or "
                  f"image: {type(exc).__name__}: {exc}")
        return None

    # ------------------------------------------------------------ segmentation
    def _selected_model_path(self) -> Optional[str]:
        model_path = self.model_paths.get(self.model_selector.value)
        if not model_path:
            print("Segmentation -> No model selected")
            return None
        return model_path

    def load_model(self, x=None) -> None:
        """Eagerly load the selected checkpoint (old load_model)."""
        model_path = self._selected_model_path()
        if model_path is None:
            self.model_segmenter = None
            self.model_checkpoint = None
            return
        try:
            self.model_segmenter = SingleModelSegmenter(model_path).load()
            self.model_checkpoint = model_path
            print(f"{get_time_now()} Segmentation -> Model loaded: "
                  f"{self.model_selector.value} ({model_path})")
        except Exception as exc:
            self.model_segmenter = None
            self.model_checkpoint = None
            show_error_message(f"An error occurred while loading the model: {exc}")
            print(f"{get_time_now()} Segmentation -> Error loading model "
                  f"{self.model_selector.value}: {type(exc).__name__}: {exc}")

    def evaluate_image(self, x=None) -> None:
        """Run the selected model on the current time slice (old evaluate_image)."""
        if self.napri_cell_image is None:
            print("Segmentation -> No image loaded")
            return None
        # predict parameters
        diameter = self.model_pars_diameter.value
        cellprob_threshold = self.model_pars_cellprob.value
        flow_threshold = self.model_pars_flow.value
        niter = self.model_pars_niter.value
        # get the current image (layer-local) and its source time-point
        time_index = int(self.viewer.dims.point[0])
        current_image = self.napri_cell_image.data[time_index, ...]
        source_index = self.time_points_slice.start + time_index
        source_time = self.time_points[source_index]
        if self.optical_missing_frames[self.cell_detection_opticalChannel_name][source_index]:
            print(f"Segmentation -> {self.cell_detection_opticalChannel_name!r} has no "
                  f"image at {source_time!r}; displayed zeros only, model not run")
            return None
        # build the segmenter lazily for the selected checkpoint
        model_path = self._selected_model_path()
        if model_path is None:
            return None
        if self.model_segmenter is None or self.model_checkpoint != model_path:
            print(f"{get_time_now()} Segmentation -> Load model {model_path}")
            self.model_segmenter = SingleModelSegmenter(model_path)
            self.model_checkpoint = model_path
        print("Segmentation -> Run model")
        try:
            masks = self.model_segmenter.eval(
                current_image, diameter=float(diameter),
                flow_threshold=float(flow_threshold),
                cellprob_threshold=float(cellprob_threshold),
                niter=int(niter))
        except Exception as exc:
            show_error_message(f"An error occurred while running the model: {exc}")
            print(f"{get_time_now()} Segmentation -> Error running model: "
                  f"{type(exc).__name__}: {exc}")
            return None
        # Keep live edits when switching channels or reloading the cropped layers.
        # This is in-memory only; Save writes a TIFF and does not alter the pkl.
        self.cells_mask_stack[source_index] = masks
        self.fov_cells_mask_dict.setdefault(self.channel_index, {})[source_time] = masks
        self.napri_cell_mask.data[time_index, ...] = masks
        self.napri_cell_mask.refresh()
        return None

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        print(f"{get_time_now()} moma_pipeline FOV viewer: Initializing Napari")
        napari.run()


def build_parser() -> argparse.ArgumentParser:
    """CLI parser for the moma-view-fov console script."""
    parser = argparse.ArgumentParser(
        prog="moma-view-fov",
        description="napari viewer for MoMA exported FOV data (DT pickles).")
    parser.add_argument(
        "exported_data_dir", nargs="?", default=None,
        help="directory containing *exported_data.pkl files "
             "(default: current working directory)")
    parser.add_argument(
        "--model_dir", "--model-dir", action="append", default=None,
        metavar="DIR", help="extra directory with model checkpoints; "
                            "repeatable (one level of subfolders is scanned)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    exported_data_dir = args.exported_data_dir or os.getcwd()
    viewer = FovViewer(exported_data_dir=exported_data_dir,
                       model_dirs=args.model_dir)
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
