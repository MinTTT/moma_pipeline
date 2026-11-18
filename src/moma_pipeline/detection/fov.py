# -*- coding: utf-8 -*-
"""
Segregating the channels from the mother machine images.

Merged DT-native MomaFOV class, combining the old Moma_CPM_FOV.MomaFOV
(detection/cropping) with moma_dt_pipeline_utils.MomaFOV_DT (DT morphology
calculation and export). Model inference is delegated to the lazy
moma_pipeline.models engine (MoMa_CPM / CPM checkpoint constants / torch
probing were removed; see deletions below).

@author:  CHU Pan
@data: 2024-08-11
@e-mail: pan_chu@outlook.com


data structure of dictionaries:

channel_keys: list
    keys of the side channels

multi_optical_channels_time_points: {time_point: {OpticalChannel: image name}}
    time_point: str
        key of the time point
    OpticalChannel: str
        key of the channel
    image name: str
        name of the image


multi_optical_channels_time_points_path: {time_point: {OpticalChannel: image_path}}
    time_point: str
        key of the time point
    OpticalChannel: str
        key of the channel
    image_path: str
        path of the image

fov_cells_mask_dict: {channel_key: {time_key: cells_mask}}
    channel_key: int
        key of the channel
    time_key: str
        key of the time point
    cells_mask: np.darray
        mask of cells in the channel

fov_channels_image_dict: {channel_key: {time_key: {channel: image}}}
    channel_key: int
        key of the channel
    time_key: str
        key of the time point
    channel: str
        key of the channel
    image: np.ndarray
        image of the channel

fov_cells_dict: {channel_key: {time_key: [cell]}}
    
    channel_key: int
        key of the channel
    time_key: str
        key of the time point
    cell: CellDT
        cell object

"""

#%%
import enum
import os
import pickle

import numpy as np
from skimage import io
from typing import List, Dict, Optional
from joblib import Parallel, delayed
from tqdm import tqdm

from moma_pipeline.core.logging import LogRecorder, SafeLogRecorder  # noqa: F401
from moma_pipeline.core.fov_utils import get_time_now
from moma_pipeline.io.images import (read_image, driftcorr, cropbox,
                                     getDriftTemplate, getChamberBoxes_CPM,
                                     getBoundingRect, filter_small_cells)
from moma_pipeline.io.contours import cv_find_contours  # noqa: F401
from moma_pipeline.io.dt_exports import export_cells_to_dataframe_dt
from moma_pipeline.io.fov_pkl import (export_Moma_fov_data_dt,
                                      save_flu_distribution_dt)

DEFAULT_MAX_NUM_CELLS_ANALYZED = 4  # maximum number of cells analyzed in a channel, if more than this number of cells are detected, the channel will be considered as a filled channel and will be removed.


def determine_channel_direction(channel_phase_image: np.ndarray,
                                geometric_center_y: float,
                                bin_size: int = 15) -> bool:
    """Determine outlet polarity from binned brightness and channel-box geometry.

    Parameters
    ----------
    channel_phase_image : np.ndarray
        Nonempty, finite 2D image of one channel row, including outlet context.
    geometric_center_y : float
        Median channel-box midpoint, minus this image crop's starting Y.
        Coordinates refer to pixel edges, matching half-open box/crop bounds.
    bin_size : int, optional
        Number of neighboring Y rows averaged per bin, by default 15 pixels.
        The final shorter bin uses its actual height.

    Returns
    -------
    bool
        True if the outlet is upward in the original image; False if downward.

    Raises
    ------
    ValueError
        If inputs are invalid, a brightest bin straddles the reference, or
        equally bright bins support opposite directions.
    """
    image = np.asarray(channel_phase_image)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("channel_phase_image must be a nonempty 2D image")
    if not np.isfinite(geometric_center_y) or not 0 < geometric_center_y < image.shape[0]:
        raise ValueError("geometric_center_y must lie inside the cropped image")
    if (isinstance(bin_size, (bool, np.bool_))
            or not isinstance(bin_size, (int, np.integer)) or bin_size <= 0):
        raise ValueError("bin_size must be a positive integer")

    mean_vertical = np.mean(image, axis=1, dtype=np.float64)
    if not np.all(np.isfinite(mean_vertical)):
        raise ValueError("channel_phase_image must contain only finite intensities")
    starts = np.arange(0, image.shape[0], bin_size)
    stops = np.minimum(starts + bin_size, image.shape[0])
    binned_means = np.add.reduceat(mean_vertical, starts) / (stops - starts)
    brightest = np.flatnonzero(binned_means == binned_means.max())
    if np.any((starts[brightest] < geometric_center_y)
              & (geometric_center_y < stops[brightest])):
        raise ValueError("Brightest bin straddles the geometric reference; direction is ambiguous")
    outlet_up = (starts[brightest] + stops[brightest]) / 2 < geometric_center_y
    if np.any(outlet_up != outlet_up[0]):
        raise ValueError("Equally bright bins on opposite sides; direction is ambiguous")
    return bool(outlet_up[0])


class MomaFOV:
    """
    This class defines parameters of a FOV

    Attributes
    ----------
    fov_directory : str
        the directory of the FOV
    cell_detection_opticalChannel_name : str
        the name of the optical channel used for cell detection
    channel_model_path : str
        the path of the channel model
    cell_model_path : str
        the path of the cell model

    time_points

    multi_optical_channels_time_points
    multi_optical_channels_time_points_path
    """

    def __init__(self, fov_directory: str,
                 cell_detection_opticalChannel_name: str,
                 MoMa_CP_model: Optional[object] = None,
                 cell_model_params: Optional[Dict] = None,
                 channel_model_params: Optional[Dict] = None,
                 log_recorder: Optional[LogRecorder] = None,
                 max_cell_number: int = DEFAULT_MAX_NUM_CELLS_ANALYZED,
                 dt_jobs: int = -1,
                 flu_distribution: bool = False,
                 flu_distribution_channels: Optional[List[str]] = None,
                 flu_distribution_bins: int = 15,
                 keep_models_loaded: bool = False):
        """

        Parameters
        ----------
        fov_directory : str
           directory for fov
        cell_detection_opticalChannel_name :  str
            Optical channel for cell detection
        MoMa_CP_model : object, optional
            A model engine for cell and channel detection (duck-typed with the
            old MoMa_CPM: load_models/release_models/eval_images/describe_models,
            eval_cell_params/eval_channel_params, cell_model_path/channel_model_path).
            If None, a new engine is created lazily via
            moma_pipeline.models.create_engine.
        cell_model_params : dict, optional
            Parameters for the cell model. If None, default parameters will be used.
        channel_model_params : dict, optional
            Parameters for the channel model. If None, default parameters will be used.
        log_recorder : LogRecorder, optional
            Recorder for log messages. If None, a console-only LogRecorder is created.
        max_cell_number : int, optional
            maximum number of cells analyzed in a channel per time point.
        dt_jobs : int, optional
            Total number of concurrent DT morphology workers (joblib loky pool).
        flu_distribution : bool, optional
            Export per-cell binned fluorescence distribution pickles.
        flu_distribution_channels : list of str, optional
            Channels used for the fluorescence distribution; None selects all
            non-cell-detection channels available per cropped channel/timepoint.
        flu_distribution_bins : int, optional
            Number of bins for the fluorescence distribution.
        keep_models_loaded : bool, optional
            Keep channel and cell models resident across FOVs. If False, the
            channel model is released after channel detection and the cell
            model after time-series cell detection.

        Returns
        -------

        """

        # DT morphology/export options (ported from old MomaFOV_DT.__init__)
        self.max_cell_number = max_cell_number
        self.dt_jobs = dt_jobs
        self.flu_distribution = bool(flu_distribution)
        self.flu_distribution_channels = flu_distribution_channels
        self.flu_distribution_bins = flu_distribution_bins

        self.cell_model_params = cell_model_params
        self.channel_model_params = channel_model_params
        self.keep_models_loaded = bool(keep_models_loaded)
        self.fov_directory = fov_directory
        self.cell_detection_opticalChannel_name = cell_detection_opticalChannel_name

        self.fov_name = os.path.basename(fov_directory)
        self.fov_dir_name = os.path.dirname(fov_directory)
        self.export_dir = os.path.join(self.fov_dir_name, 'exported_data')
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

        # 1. log recorder
        if log_recorder is None:
            self.log_recorder = LogRecorder()  # print log to console only
        else:
            self.log_recorder = log_recorder

        # 12. find all optical channels in the FOV directory, it is the folder name in the FOV directory
        self.opticalChannels = [file for file in os.listdir(fov_directory)
                                if os.path.isdir(os.path.join(fov_directory, file))]

        if cell_detection_opticalChannel_name in self.opticalChannels:
            self.log_recorder.write_log(f'Moma_CPM -> {self.fov_name}: cell-detection optical channel is in existing optical channels')
        else:
            exception_str = f'{get_time_now()} Moma_CPM -> {self.fov_name} cell detection optical channel is not in existing optical channels'
            raise Exception(exception_str)



        """ ====================== MoMa images fetch related ===================
        self.cell_detection_opticalChannel_name: str
            Name of the optical channel used for cell detection.
        self.detection_images_path: List[str]
            List of paths to the images used for cell detection in the specified optical channel.
        self.cell_detect_opticalChannel_images_name_list: List[str]
            List of image names in the specified optical channel used for cell detection.
        self.other_channels: List[str]
            List of other optical channels available in the FOV directory, excluding the cell detection channel.
        self.other_channels_image_dict: Dict[str, Dict[str, str]]
            Dictionary containing paths to images for each optical channel, excluding the cell detection channel. {Optical_channel_name: {image_name: image_path}}
        self.time_points: List[str]
            List of time points available in the cell detection optical channel. It is sorted based on the time point names, and in a increasing order.
        self.multi_optical_channels_time_points: Dict[str, Dict[str, str]]
            Dictionary containing time points and their corresponding images for each optical channel. {time_point: {OpticalChannel: image name}}
        self.multi_optical_channels_time_points_path: Dict[str, Dict[str, str]]
            Dictionary containing time points and their corresponding image paths for each optical channel. {time_point: {OpticalChannel: image_path}}
        
        """
        self.cell_detect_opticalChannel_images_name_list = [file
                                                            for file in os.listdir(
                os.path.join(fov_directory, self.cell_detection_opticalChannel_name))
                                                            if file.split('.')[-1] in 'tiffpng']
        # self.detection_images_path = [os.path.join(fov_directory, cell_detection_opticalChannel_name, file)
        #                               for file in self.cell_detect_opticalChannel_images_name_list]
        self.detection_opticalChannel_image_dict = {
            image_name: os.path.join(fov_directory, self.cell_detection_opticalChannel_name, image_name)
            for image_name in self.cell_detect_opticalChannel_images_name_list}
        self.other_channels = [channel for channel in self.opticalChannels
                               if channel != self.cell_detection_opticalChannel_name]
        # self.other_channels_image_dict will store images path, along the order:
        # {Optical_channel_name: {image_name: image_path}}
        self.other_channels_image_dict = {channel: {} for channel in
                                          self.other_channels}  # {channel: {image_name: image_path}}
        for channel in self.other_channels:
            channel_images_name = [file for file in os.listdir(os.path.join(fov_directory, channel))
                                   if file.split('.')[-1] in 'tiffpng']
            channel_images_path = [os.path.join(fov_directory, channel, file)
                                   for file in channel_images_name]
            for i in range(len(channel_images_name)):
                self.other_channels_image_dict[channel][channel_images_name[i]] = channel_images_path[i]

        self.time_points = list(self.detection_opticalChannel_image_dict.keys())

        time_point = self.time_points[0]
        if len(time_point.split('.')[0].split('_')) == 5:
            # sort the time points
            self.time_points.sort(key=lambda elem: int(elem.split('.')[0].split('_')[-1]))
            self.data_structure_version = '0.2'
        else:
            self.time_points.sort(key=lambda elem: int(elem.split('.')[0].strip('t')))
            self.data_structure_version = '0.1'

        self.multi_optical_channels_time_points = {time_point: {} for time_point in self.time_points}
        self.multi_optical_channels_time_points_path = {time_point: {} for time_point in self.time_points}
        self.find_multi_optical_channels_time_points()
        # 1.2 summarize the FOV information
        self.log_recorder.write_log(f'{get_time_now()} Moma_CPM -> {self.fov_name}: ')
        self.log_recorder.write_log(f'    {len(self.opticalChannels)} optical channels found: {self.opticalChannels}')
        self.log_recorder.write_log(
            f'    {len(self.cell_detect_opticalChannel_images_name_list)} images in the cell detection optical channel: {self.cell_detection_opticalChannel_name}')

        # 2. load models
        '''
        self.cell_model: models.CellposeModel
        
        '''
        self.cell_model_params = cell_model_params if cell_model_params is not None else {}
        self.channel_model_params = channel_model_params if channel_model_params is not None else {}
        self.channel_size_threshold_factor = 0.8
        self.cells_fill_threshold = 2  # type is int # the minimum number of cells in a channel to be considered filled, only channels with no less than this number of cells will be kept
        if MoMa_CP_model is not None:
            self.MoMa_CP_model = MoMa_CP_model
        else:
            self.log_recorder.write_log(f'{get_time_now()} Moma_CPM -> No existing MoMa_CP_model provided, creating a new one by default '
                  f'parameters.')
            # Imported function-level so this module stays importable without
            # torch/cellpose/omnishim (moma_pipeline.models resolves lazily).
            from moma_pipeline.models import create_engine
            self.MoMa_CP_model = create_engine(
                cell_model_params=self.cell_model_params,
                channel_model_params=self.channel_model_params,
                log_recorder=self.log_recorder)
        # Get model paths from the engine (duck-typed with the old MoMa_CPM)
        self.cell_model_path = self.MoMa_CP_model.cell_model_path
        self.channel_model_path = self.MoMa_CP_model.channel_model_path

        # 3. Initialize attributes for channel detection and cell detection
        self.flip_flag = None  # type is Optional[bool]
        self.channel_mask = None  # type is Optional[np.ndarray]
        self.channel_template = None  # type is Optional[np.ndarray]
        self.channel_detection_image = None  # type is Optional[np.ndarray]
        self.channel_boxes = {}  # type is Dict[int, Dict[str, int]]
        self.channel_index = []  # type is List[int]
        self.channel_directions = None

    def find_multi_optical_channels_time_points(self):
        """
        Find multi-optical channels time points and store the image names and paths.
        In this method, we will create two dictionaries, which will be used to store the time points and their
        corresponding images for each optical channel:
        multi_optical_channels_time_points: Dict[str, Dict[str, str]]
            Dictionary containing time points and their corresponding images for each optical channel.
            {time_point: {OpticalChannel: image name}}
        multi_optical_channels_time_points_path: Dict[str, Dict[str, str]]
            Dictionary containing time points and their corresponding image paths for each optical channel.
            {time_point: {OpticalChannel: image_path}}
        """

        for time_point in self.time_points:
            self.multi_optical_channels_time_points[time_point][self.cell_detection_opticalChannel_name] = time_point
            self.multi_optical_channels_time_points_path[time_point][self.cell_detection_opticalChannel_name] = \
                self.detection_opticalChannel_image_dict[time_point]
            for channel in self.other_channels:
                image_name = time_point.replace(self.cell_detection_opticalChannel_name, channel)
                if image_name in self.other_channels_image_dict[channel]:
                    self.multi_optical_channels_time_points[time_point][channel] = image_name
                    self.multi_optical_channels_time_points_path[time_point][channel] = \
                        self.other_channels_image_dict[channel][image_name]

    def get_images(self, image_index: int = 0, optical_channel: Optional[str] = None,
                   flip_flag: bool = False):
        """
        get the channel or cell detection image for CPModel
        Parameters
        ----------
        channel :
        image_index : int
            the index of the channel detection image in the list of cell detection images
            by default it is 0, which means the first image in the list.

        flip_flag : bool
            if True, the image will be flipped vertically, by default it is False.
            This is useful when the channel detection image is flipped vertically.

        Returns
        -------
        channel_detection_image : np.ndarray
            the channel detection image
        """
        time_point = self.time_points[image_index]

        if optical_channel is None:
            # if optical_channel is not specified, use the cell detection optical channel
            image_path = self.multi_optical_channels_time_points_path[time_point][
                self.cell_detection_opticalChannel_name]
        else:
            # if optical_channel is specified, use the specified channel
            # check if the optical_channel is in the other_channels_image_dict
            if optical_channel in self.other_channels:
                image_path = self.multi_optical_channels_time_points_path[time_point][optical_channel]
            else:
                error_msg = f"{get_time_now()} MoMa FOV -> Optical channel '{optical_channel}' is not in the other channels: {self.other_channels}"
                self.log_recorder.write_log(error_msg)
                raise ValueError(error_msg)

        image = io.imread(image_path)
        if flip_flag:
            image = image[::-1, :]

        return image


    def detect_and_crop_channels(self, channel_mask_save: bool = True):
        """
        detect channels (side grooves) in the FOV.
        Parameters
        ---------
        channel_mask_save : bool
            if True, the channel mask will be saved to the export directory, by default it is True.
            The channel mask will be saved as a tif file with the name '{fov_name}_channel_mask.tif'.

        """
        # 1. find all channels in the FOV
        # 1.a load channel detection model and images
        channel_detection_image = self.get_images(image_index=0)
        try:
            channel_mask = self.MoMa_CP_model.eval_images(
                channel_detection_image, model='channel')[0]
        finally:
            self._release_model_after_detection_phase('channel')
        if channel_mask_save:
            # save the channel mask to the export directory
            time_now = get_time_now('2')
            channel_mask_save_path = os.path.join(self.export_dir, f'{self.fov_name}_{time_now}_channel_masks.tif')
            channel_image_save_path = os.path.join(self.export_dir, f'{self.fov_name}_{time_now}_channel.tif')
            io.imsave(channel_mask_save_path, channel_mask.astype(np.uint16), check_contrast=False)
            io.imsave(channel_image_save_path, channel_detection_image, check_contrast=False)
            self.log_recorder.write_log(f'{get_time_now()} Moma_CPM -> {self.fov_name}: Channel mask saved to {channel_mask_save_path}')
        channel_numbers = channel_mask.max().astype(int)  # type is int
        # check if channel_number is 0
        if channel_numbers == 0:
            return 1  # no channels detected, return 1 to indicate no channels found

        # 2. get the channel boxes
        channel_boxes_list = getBoundingRect(channel_mask)  # get all boxes
        channel_box_value = list(range(1, len(channel_boxes_list)+1))
        # check the boxes morphology, if the width or height is less of greater too much of the 0.9 quantile value, remove it
        channel_height_9quantile = np.quantile([abs(box['ybr'] - box['ytl']) for box in channel_boxes_list], 0.90)
        channel_width_9quantile = np.quantile([abs(box['xbr'] - box['xtl']) for box in channel_boxes_list], 0.90)
        print(f'Channel width 90% quantile: {channel_width_9quantile}')
        print(f'Channel height 90% quantile: {channel_height_9quantile}')
        remove_num = 0
        channel_num = len(channel_boxes_list)
        valid_channel_list =[]
        valid_channel_value = []
        for box_i, box in enumerate(channel_boxes_list):
            box_width = abs(box['xbr'] - box['xtl'])
            box_height = abs(box['ybr'] - box['ytl'])
            if box_width < channel_width_9quantile * 0.4 or box_width > channel_width_9quantile * 1.2 or \
                    box_height < channel_height_9quantile * 0.4 or box_height > channel_height_9quantile * 1.2:
                remove_num += 1
            else:
                valid_channel_list.append(box)
                valid_channel_value.append(channel_box_value[box_i])
        print(f'{remove_num}/{channel_num} channels removed due to their morphology.')
        self.log_recorder.write_log(f'{get_time_now()} Moma_CPM -> {self.fov_name}: {len(valid_channel_list)} channels detected.')
        # 3. determine the direction of the channel
        # for one FOV, there may have multiple rows of channels, and they may have different directions.
        # 3.a determine the rows of channels, and determine the direction of each row
        rows_boxes_y_center = {}  # type is Dict[int, List[int]] ## it stores the y center of the boxes in each row
        rows_boxes = {}  # type is Dict[int, List[Dict[str, int]]] ## it stores the boxes bounding box in each row
        rows_boxes_mask_values = {}  # type is Dict[int, List[int]] ## it stores the boxes mask values in each row
        for box_i, box in enumerate(valid_channel_list):
            box_y_center = (box['ytl'] + box['ybr']) // 2
            box_height = abs(box['ybr'] - box['ytl'])
            if len(rows_boxes_y_center) == 0:
                rows_boxes_y_center[1] = [box_y_center]
                rows_boxes[1] = [box]
                rows_boxes_mask_values[1] = [valid_channel_value[box_i]]
            else:
                # if a new box center is close to the mean of the existing rows, add it to the existing row
                match_row_flag = False
                for row_index in rows_boxes_y_center:
                    row_mean = np.mean(rows_boxes_y_center[row_index])
                    if abs(row_mean - box_y_center) <= box_height * 0.8:  # 80% of the box height
                        rows_boxes_y_center[row_index].append(box_y_center)
                        rows_boxes[row_index].append(box)
                        rows_boxes_mask_values[row_index].append(valid_channel_value[box_i])
                        match_row_flag = True
                        break
                    else:
                        pass  # when code runs here, it means that the box is not close to current row, so check the
                        # next row
                # when code runs here, it means that the box is not close to any existing row, so create a new row
                if not match_row_flag:
                    rows_boxes_y_center[len(rows_boxes_y_center) + 1] = [box_y_center]
                    rows_boxes[len(rows_boxes) + 1] = [box]
                    rows_boxes_mask_values[len(rows_boxes_mask_values) + 1] = [valid_channel_value[box_i]]
        self.log_recorder.write_log(f'{get_time_now()} Moma_CPM -> {self.fov_name}: {len(rows_boxes_y_center)} rows of channels detected.')
        for row_index in rows_boxes_y_center:
            self.log_recorder.write_log(f'    Row {row_index}: {len(rows_boxes_y_center[row_index])} boxes detected, '
                  f'boxes mask values: {rows_boxes_mask_values[row_index]}')
        # save channel info to a text file: columns: fov_name, row_index, boxes_mask_values, channel_index, channel_length, channel_width
        channel_info_path = os.path.join(self.export_dir, f'channel_info.txt')
        if not os.path.exists(channel_info_path):
            with open(channel_info_path, 'w') as f:
                f.write('fov_name\trow_index\tboxes_mask_values\tchannel_index\tchannel_length\tchannel_width\n')
                # if the file exists, append the new data
        if len(rows_boxes_y_center) > 0:
            with open(channel_info_path, 'a') as f:
                for row_index in rows_boxes_y_center:
                    boxes_mask_values = ', '.join([str(vl) for vl in rows_boxes_mask_values[row_index]])
                    for box_i, box in enumerate(rows_boxes[row_index]):
                        channel_width = abs(box['xbr'] - box['xtl'])
                        channel_length = abs(box['ybr'] - box['ytl'])
                        f.write(f'{self.fov_name}\t{row_index}\t{boxes_mask_values}\t{box_i+1}\t{channel_length}\t{channel_width}\n')

        # 3.b determine the direction of each row
        channel_directions = {}  # type is Dict[int, bool]
        for row_index in rows_boxes_y_center:
            boxes_in_row = rows_boxes[row_index]
            boxes_mask_values_in_row = rows_boxes_mask_values[row_index]
            row_top_left_y = int(min([box['ytl'] for box in boxes_in_row]))
            row_bottom_right_y = int(max([box['ybr'] for box in boxes_in_row]))
            row_dist = row_bottom_right_y - row_top_left_y
            row_top_left_bound = int(row_top_left_y - row_dist * 0.1)  # 10% of the row height
            row_top_left_bound = max(row_top_left_bound, 0)  # ensure the bound is not negative
            row_bottom_right_bound = int(row_bottom_right_y + row_dist * 0.1)  # 10% of the row height
            row_bottom_right_bound = min(row_bottom_right_bound, channel_detection_image.shape[0])
            row_detection_image = channel_detection_image[row_top_left_bound:row_bottom_right_bound, ...]
            geometric_center_y = float(np.median([
                (box['ytl'] + box['ybr']) / 2 for box in boxes_in_row
            ])) - row_top_left_bound
            flip_flag = determine_channel_direction(
                row_detection_image, geometric_center_y=geometric_center_y)
            for box_mask_vlue in boxes_mask_values_in_row:
                channel_directions[box_mask_vlue] = flip_flag
        # 4. crop the channels from the channel detection image
        # Note: 1. channels_image; 2. channel_boxes; 3. channel_directions
        # channels_image stores the cropped channel images (dead end towards image top),
        # channel_boxes stores the boxes of the channels,
        # channel_directions stores the direction of each channel, True means the outlet is upward in original image,
        # False means the outlet is downward.
        # 4.a get the channel boxes and crop the channels from the channel detection image
        channel_boxes = {}  # type is Dict[int, Dict[str, int]]
        for mask_value in channel_directions:
            single_channel_mask = channel_mask == mask_value
            channel_boxes[mask_value] = getChamberBoxes_CPM(single_channel_mask, flip=channel_directions[mask_value])[0]

        channels_image = {}  # type is Dict[int, np.ndarray]

        self.log_recorder.write_log(f'{get_time_now()} Moma_CPM -> {self.fov_name}: {len(channel_boxes)} channel box info:')
        # 4.b crop the channels from the channel detection image. Get channels_image and store them in a dictionary
        for box_index in channel_boxes:
            # print the channel box info:
            self.log_recorder.write_log(f'    Channel {box_index}: {channel_boxes[box_index]}')
            channel_image = cropbox(channel_detection_image, channel_boxes[box_index])
            if channel_directions[box_index]:
                channel_image = channel_image[::-1, :]
            channels_image[box_index] = channel_image
        # 4.c if the box is close to the image border, remove it
        too_close_boxes = []  # type is List[int]
        for box_index in channel_boxes:
            box = channel_boxes[box_index]
            if box['ytl'] <= 0 or box['ybr'] >= channel_detection_image.shape[0]:
                self.log_recorder.write_log(f'{get_time_now()} Moma_CPM -> {self.fov_name}: Channel {box_index} is close to the image border, '
                      f'removing it.')
                too_close_boxes.append(box_index)
        for box_index in too_close_boxes:
            del channel_boxes[box_index]
            del channels_image[box_index]
            del channel_directions[box_index]

        channels_image_list = list(channels_image.values())
        cells_mask = self.MoMa_CP_model.eval_images(channels_image_list, model='cell')[0]


        # save all cells masks and channel images
        if channel_mask_save:
            for i, box_index in enumerate(channels_image):
                channel_mask = cells_mask[i]
                channel_image = channels_image[box_index]
                if channel_mask.max() > 0:  # if there are cells in the channel
                    time_now = get_time_now('2')
                    io.imsave(os.path.join(self.export_dir, f'{self.fov_name}_{time_now}_{box_index}_masks.tif'),
                              channel_mask, check_contrast=False)
                    io.imsave(os.path.join(self.export_dir, f'{self.fov_name}_{time_now}_{box_index}.tif'),
                              channel_image, check_contrast=False)
        cells_number_each_channel = {}
        channels_fill_cells_flag = {}
        for i, box_index in enumerate(channels_image):
            cells_number_each_channel[box_index] = int(cells_mask[i].max())
            channels_fill_cells_flag[box_index] = True if cells_number_each_channel[box_index] >= self.cells_fill_threshold else False


        filled_channels = {}
        for box_index in channels_fill_cells_flag:
            if channels_fill_cells_flag[box_index]:
                filled_channels[box_index] = channel_boxes[box_index]
        # channel_index_filled = [channel_index[i] for i in channels_fill_cells_flag if channels_fill_cells_flag[i]]
        self.channel_boxes = filled_channels
        self.channel_directions = channel_directions
        self.channel_index = list(filled_channels.keys())  # type is List[int]

        if len(self.channel_index) == 0:
            self.log_recorder.write_log(f'{get_time_now()} Moma_CPM -> {self.fov_name}: No channels with cells detected')
            return 1

        self.log_recorder.write_log(f'{get_time_now()} Moma_CPM -> {self.fov_name}: {len(self.channel_index)} channels with cells detected')
        self.channel_detection_image = channel_detection_image
        return None

    def _release_model_after_detection_phase(self, model):
        """Release a completed phase's model in memory-saving mode."""
        if not self.keep_models_loaded:
            self.MoMa_CP_model.release_models(model=model)

    def detect_cells(self, threads_num_max=2, sifht_correction: bool = True):
        """
        detect cells in each channel

        Parameters
        ----------
        threads_num_max : int
            the maximum number of threads for parallel processing
        """

        self.log_recorder.write_log(f'{get_time_now()} Moma_CPM -> {self.fov_name}: Start cell detection')

        channel_drift_correct_template = getDriftTemplate(self.channel_boxes.values(),
                                                          self.channel_detection_image)
        drift_cor_box = dict(xtl=0, xbr=None, ytl=0,
                             ybr=max(self.channel_boxes.values(), key=lambda elem: elem['ytl'])['ytl'])

        # detect cells in each image
        self.fov_cells_mask_dict = {self.channel_index[i]: {} for i in range(len(self.channel_index))}
        self.fov_channels_image_dict = {self.channel_index[i]: {} for i in
                                        range(len(self.channel_index))}
        self.time_points_seconds = {}
        self.images_background_fluorescence = {}

        def _thread_find_cells(image_index, time_points,
                               multi_optical_channels_time_points,
                               multi_optical_channels_time_points_path,
                               cell_detection_opticalChannel_name,
                               channel_drift_correct_template, drift_cor_box,
                               flip_flag, channel_boxes, channel_index, CP_model,
                               fov_cells_mask_dict,
                               fov_channels_image_dict, time_points_seconds, images_background_fluorescence,
                               time_key, sifht_correction):
            """
            read an image and find cells
            1. read the image
            2. drift correction
            3. crop the image by channel
            4. detect cells in each channel

            """
            image_name = time_points[image_index]
            opticalChannels_images_path = multi_optical_channels_time_points_path[image_name]
            opticalChannels_images_name = multi_optical_channels_time_points[image_name]

            # read images
            images = {}

            for optical_channel in opticalChannels_images_name:
                img, time = read_image(opticalChannels_images_path[optical_channel], read_time=True, time_key=time_key)
                images[optical_channel] = img
                time_points_seconds[opticalChannels_images_name[optical_channel]] = time

            # if flip_flag:
            #     images = {channel: image[::-1, :] for channel, image in images.items()}

            # drift correction for each optical channel
            # 1. shift image
            if sifht_correction:
                image = images[cell_detection_opticalChannel_name]
                image_shifted, drift = driftcorr(image, channel_drift_correct_template, box=drift_cor_box, parallel=False)
                images[cell_detection_opticalChannel_name] = image_shifted
                other_channels = [OptChannel for OptChannel in images if OptChannel != cell_detection_opticalChannel_name]
                for OptChannel in other_channels:
                    image = images[OptChannel]
                    image_shifted, _ = driftcorr(img=image, drift=drift, parallel=False)
                    images[OptChannel] = image_shifted
            # for opticalChannel, image in images.items():
            #     image_shifted, _ = driftcorr(image, channel_drift_correct_template, box=drift_cor_box, parallel=False)
            #     images[opticalChannel] = image_shifted
            # 2. crop the image by side channel
            channels_image = []  # store the images of each side channel for cell detection
            for channel_i, box in channel_boxes.items():
                # crop all optical channel images by channel
                images_crop = {}
                for opticalChannel, image in images.items():
                    channel_image = cropbox(images[opticalChannel], box)
                    if flip_flag[channel_i]:
                        channel_image = channel_image[::-1, :]
                    images_crop[opticalChannel] = channel_image
                channels_image.append(images_crop[cell_detection_opticalChannel_name])
                fov_channels_image_dict[channel_i][image_name] = images_crop
            # 3. predict cells in each channel
            cells_in_channels = CP_model.eval_images(channels_image, model='cell')
            cells_masks_list = cells_in_channels[0]
            # 3.a filter cells that are too small
            cells_masks_list = [filter_small_cells(cells_mask) for cells_mask in cells_masks_list]
            for channel_i, cells_mask in enumerate(cells_masks_list):
                fov_cells_mask_dict[channel_index[channel_i]][image_name] = cells_mask

            # 4. calculate images fluorescence intensity of background
            image_size = images[cell_detection_opticalChannel_name].shape
            y_bottom = max([box['ybr'] for box in channel_boxes.values()])  # the bottom of the channel boxes
            bac_region_y_min = y_bottom + 50 if y_bottom + 50 < image_size[0] else y_bottom
            bac_region_ymax = y_bottom + 200 if y_bottom + 200 < image_size[0] else image_size[0]
            backgroup_reagion = np.s_[bac_region_y_min:bac_region_ymax, :]
            background_fluorescence = {channel: {} for channel in list(images.keys())}
            for channel, image in images.items():
                background_fluorescence[channel][f'mean'] = np.mean(image[backgroup_reagion])
                background_fluorescence[channel][f'median'] = np.median(image[backgroup_reagion])
            images_background_fluorescence[image_name] = background_fluorescence


            return None

        if self.data_structure_version == '0.1':
            time_key = 'time'
        elif self.data_structure_version == '0.2':
            time_key = 'acq_time'
        else:
            time_key = None

        try:
            _ = Parallel(n_jobs=threads_num_max, backend='threading',
                         require='sharedmem')(delayed(_thread_find_cells)
                                              (image_index, self.time_points,
                                               self.multi_optical_channels_time_points,
                                               self.multi_optical_channels_time_points_path,
                                               self.cell_detection_opticalChannel_name,
                                               channel_drift_correct_template, drift_cor_box,
                                               self.channel_directions, self.channel_boxes, self.channel_index,
                                               self.MoMa_CP_model, self.fov_cells_mask_dict,
                                               self.fov_channels_image_dict, self.time_points_seconds,
                                               self.images_background_fluorescence, time_key, sifht_correction
                                               )
                                              for image_index in tqdm(range(len(self.time_points)))
                                              )
        finally:
            self._release_model_after_detection_phase('cell')

        return None

    def calculate_cell_parameters(self):
        """Compute morphology using CellDT.compute_from_mask().

        All (channel, timepoint) pairs are processed by one flat loky
        process pool with at most dt_jobs concurrent workers in total.
        Tasks carry only plain picklable data (not self, which holds the
        Cellpose models and the log file handle), and computation runs in
        the module-level dt_timepoint_compute. Results stream back in
        submission order, keeping row ordering deterministic.
        """
        # Imported function-level so this module (and loky workers importing
        # the task by reference) stay free of numba until computation runs.
        from moma_pipeline.morphology.worker import dt_timepoint_compute

        channel_keys = list(self.fov_cells_mask_dict.keys())
        self.fov_cells_dict = {channel_key: {} for channel_key in channel_keys}
        dt_workers = self._resolve_dt_workers()
        tasks = [(channel_key, time_key)
                 for channel_key in channel_keys
                 for time_key in self.time_points]

        progress_bar = tqdm(
            total=len(tasks), desc=f'DT {self.fov_name}',
            leave=False, dynamic_ncols=True)

        def build_worker_args(task):
            channel_key, time_key = task
            c_images = self.fov_channels_image_dict[channel_key].get(
                time_key, {})
            return (
                self.fov_cells_mask_dict[channel_key].get(time_key),
                c_images,
                self.images_background_fluorescence.get(time_key, {}),
                self.max_cell_number,
                self._resolve_flu_distribution_channels(c_images),
                self.flu_distribution_bins,
                time_key,
            )

        results = Parallel(n_jobs=dt_workers, backend='loky',
                           return_as='generator')(
            delayed(dt_timepoint_compute)(*build_worker_args(task))
            for task in tasks
        )
        for (channel_key, time_key), cells in zip(tasks, results):
            self.fov_cells_dict[channel_key][time_key] = cells
            progress_bar.update(1)
        progress_bar.close()

        self.log_recorder.write_log(
            f'{get_time_now()} Moma_CPM_DT -> {self.fov_name}: '
            f'Cell parameters calculated for {len(channel_keys)} channels x '
            f'{len(self.time_points)} timepoints with flat loky '
            f'dt_workers={dt_workers}.'
        )

    def _resolve_dt_workers(self):
        """Resolve the total number of concurrent DT workers."""
        cpu_number = os.cpu_count() or 1
        total_jobs = int(self.dt_jobs)
        if total_jobs <= 0:
            total_jobs = cpu_number
        return max(1, total_jobs)

    def _resolve_flu_distribution_channels(self, channel_images):
        """Resolve distribution channels for one cropped channel/timepoint."""
        if not self.flu_distribution:
            return None

        available = list(channel_images.keys())
        lower_to_channel = {str(channel).lower(): channel for channel in available}
        if self.flu_distribution_channels:
            selected = []
            for channel in self.flu_distribution_channels:
                matched = lower_to_channel.get(str(channel).lower())
                if matched is not None:
                    selected.append(matched)
            return selected

        detection_channel = str(self.cell_detection_opticalChannel_name).lower()
        return [
            channel for channel in available
            if str(channel).lower() != detection_channel
        ]

    def export_cells_data(self):
        """Export CSV and pickle with DT-specific fields."""
        cells_df = export_cells_to_dataframe_dt(
            self.fov_cells_dict, self.time_points_seconds,
            self.opticalChannels, self.images_background_fluorescence,
        )
        cells_df.to_csv(
            os.path.join(self.export_dir, f'{self.fov_name}_cells_df.csv'),
            index=False,
        )

        exported_data = export_Moma_fov_data_dt(
            self.time_points, self.opticalChannels, self.other_channels,
            self.cell_detection_opticalChannel_name, self.channel_index,
            self.fov_cells_mask_dict, self.fov_channels_image_dict,
            self.fov_cells_dict, self.time_points_seconds,
            self.images_background_fluorescence,
        )
        with open(os.path.join(self.export_dir,
                               f'{self.fov_name}_exported_data.pkl'),
                  'wb') as file:
            pickle.dump(exported_data, file)

        if self.flu_distribution:
            output_path, distribution = save_flu_distribution_dt(
                self.export_dir, self.fov_name, self.fov_cells_dict)
            cell_count = sum(
                len(time_dict)
                for channel_dict in distribution.values()
                for time_dict in channel_dict.values()
            )
            self.log_recorder.write_log(
                f'{get_time_now()} Moma_CPM_DT -> {self.fov_name}: '
                f'Fluorescence distribution for {cell_count} cells exported '
                f'to {output_path}'
            )

        self.log_recorder.write_log(
            f'{get_time_now()} Moma_CPM_DT -> {self.fov_name}: '
            f'Data exported to {self.export_dir}'
        )
