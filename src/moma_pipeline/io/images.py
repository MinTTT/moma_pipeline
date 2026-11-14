"""
This file contains functions and class definitions that are used in pipeline.py
A lot of functions are redundant from data.py, but we keep the files separate
to minimize the risk of unforeseen bugs.

Ported from Moma_cellPose/utilities.py into the moma_pipeline package
(moma_pipeline.io.images).

@author: jblugagne
@modified by: CHU Pan

"""
from typing import Union, Iterable, Dict, List, Optional

import cv2
import numpy as np
import os
import re
import tifffile as tiff
from joblib import Parallel, delayed
from skimage import io

# Imported here and re-exported so that old call sites keep working
# (from moma_pipeline.io.images import get_time_now).
from ..core.fov_utils import get_time_now  # noqa: F401


def calculate_weighted_center(signal: Union[list, np.ndarray]) -> int:
    """
    calculate the weighted center of the signal

    

    Parameters
    -----------
    signal: array-like or list
        1D array

    Returns
    ---------------
    center_index: int

    Examples
    -----------------
    >>> sig = np.random.random(100)
        calculate_weighted_center(sig)
    >>> 50  # the center index of the signal, value may vary near 50.


    """
    signal = np.array(signal)
    signal = signal - signal.min()
    signal = (signal + 1e-4) / (signal.max() + 1e-4)
    signal = signal / signal.sum()
    integeral_signal = np.cumsum(signal)
    max_integral = integeral_signal.max()
    center_index = np.argmax(integeral_signal > max_integral / 2)
    return center_index


def read_image(file_path, read_time: bool = False,
               time_key: str = 'acq_time'):
    """Read image from file.

    Parameters
    ----------
    file_path : str
        Path to the image file.
    read_time : bool, optional
        Flag to read timepoints from filename (default is False).
    time_key : str, optional
        Key to use when reading time from metadata (default is 'acq_time').

    Returns
    -------
    numpy.ndarray or tuple
        If read_time is False, returns a 2D or 3D numpy array containing image data.
        If read_time is True, returns a tuple containing:
            - numpy.ndarray : 2D or 3D image data
            - float : Time point of the image

    Notes
    -----
    - Supports TIFF and other common image formats (via skimage.io.imread)
    - For TIFF files with read_time=True, attempts to read metadata in this order:
        1. Shaped metadata
        2. ImageJ metadata
    - The time value is extracted from the last comma-separated value in the metadata field
    - Mate data example:
            ImageJ=1.11a
            images=1
            hyperstack=true
            mode=grayscale
            acq_time=2025-06-19-03:30:51,1750275051.7853189
    """

    _, file_extension = os.path.splitext(file_path)
    if (file_extension == '.tif' or file_extension == '.tiff') and not read_time:
        img = tiff.imread(file_path)
    else:
        img = io.imread(file_path)
    if read_time:
        # read meta from tiff file
        with tiff.TiffFile(file_path) as tif_file:
            img = tif_file.asarray()

            meta_data = tif_file.shaped_metadata  # try read shaped metadat first
            if not meta_data:
                meta_data = tif_file.imagej_metadata
            else:
                meta_data = meta_data[0]

            time = meta_data[time_key]
            try:
                time = float(time.split(',')[-1])
            except AttributeError:
                pass

        return img, time

    else:
        return img


def deskew(image):
    """
    Compute rotation angle of chambers in image for rotation correction.
    From: https://gist.github.com/panzerama/beebb12a1f9f61e1a7aa8233791bc253
    Not extensively tested. You can skip rotation correction if your chambers
    are about +/- 1 degrees of horizontal

    Parameters
    ----------
    image : 2D numpy array
        Input image.

    Returns
    -------
    rotation_number : float
        Rotation angle of the chambers for correction.

    """

    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    # canny edges in scikit-image
    edges = canny(image)

    # hough lines
    hough_lines = probabilistic_hough_line(edges)

    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 0 for (x1, y1), (x2, y2) in hough_lines]

    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]

    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=180)

    # correcting for 'sideways' alignments
    rotation_number = histo[1][np.argmax(histo[0])]

    if rotation_number > 45:
        rotation_number = -(90 - rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)

    return rotation_number


def cropbox(img: np.ndarray, box: dict) -> np.ndarray:
    """
    Crop image  img[box['ytl']:box['ybr'], box['xtl']:box['xbr']]

    Parameters
    ----------
    img : np.ndarray
        Image to crop.
    box : Dict
        Dictionary describing the box to cut out, containing the following
        elements:
            - 'xtl': Top-left corner X coordinate.
            - 'ytl': Top-left corner Y coordinate.
            - 'xbr': Bottom-right corner X coordinate.
            - 'ybr': Bottom-right corner Y coordinate.

    Returns
    -------
    2D numpy array
        Cropped-out region.

    """
    cropped_image = img[box['ytl']:box['ybr'], box['xtl']:box['xbr']]
    cropped_image = np.copy(cropped_image)
    return cropped_image


def rangescale(frame, rescale):
    """
    Rescale image values to be within range

    Parameters
    ----------
    frame : 2D numpy array of uint8/uint16/float/bool
        Input image.
    rescale : Tuple of 2 values
        Values range for the rescaled image.

    Returns
    -------
    2D numpy array of floats
        Rescaled image
    """
    frame = frame.astype(np.float32)
    if np.ptp(frame) > 0:
        frame = ((frame - np.min(frame)) / np.ptp(frame)) * np.ptp(rescale) + rescale[0]
    else:
        frame = np.ones_like(frame) * (rescale[0] + rescale[1]) / 2
    return frame


def driftcorr(img: np.ndarray, template=None, box=None, drift=None, parallel=True,
              n_jobs: int = 32):
    '''
    Compute drift between current frame and the reference, and return corrected
    image

    Parameters
    ----------
    img : 2D or 3D numpy array of uint8/uint16/floats
        The frames to correct drift for.
    template : None or 2D numpy array of uint8/uint16/floats, optional
        The template for drift correction (see getDriftTemplate()).
        default is None.
    box : None or dictionary, optional
        A cropping box to extract the part of the frame to compute drift 
        correction over (see cropbox()).
        default is None.
    drift : None or tuple of 2 numpy arrays, optional
        Pre-computed drift to apply to the img stack. If this is None, you must
        provide a template and box.
        default it None.
    parallel : bool, optional
        If True, use joblib Parallel(require='sharedmem') over the frames.
        default is True.
    n_jobs : int, optional
        Number of jobs for joblib Parallel when parallel=True.
        default is 32 (preserves the original hardcoded value).

    Returns
    -------
    2D/3D numpy array, tuple of len 2
        Drift-corrected image and drift.

    '''

    if len(img.shape) == 2:
        twoDflag = True
        img = np.expand_dims(img, axis=0)
    else:
        twoDflag = False

    if drift is None:
        if template is None:  # If we have a position with 0 chambers (see getDriftTemplate)
            return img, (0, 0)
        template = rangescale(template, (0, 255)).astype(np.uint8)  # Making sure its the right format
        xcorr = np.empty([img.shape[0]])
        ycorr = np.empty([img.shape[0]])
    elif twoDflag:
        (xcorr, ycorr) = ([drift[0]], [drift[1]])
    else:
        (xcorr, ycorr) = drift

    def parallel_drift(inx):
        if drift is None:
            frame = rangescale(img[inx], (0, 255)).astype(np.uint8)  # Making sure its the right format
            driftcorrimg = cropbox(frame, box)  # crop part of image for matching
            res = cv2.matchTemplate(driftcorrimg, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)  # top left corner
            ycorr[inx] = max_loc[0] - res.shape[1] / 2  # TODO: why size should over 2 ?
            xcorr[inx] = max_loc[1] - res.shape[0] / 2  # TODO: why size should over 2 ?
        T = np.float32([[1, 0, -ycorr[inx]],
                        [0, 1, -xcorr[inx]]])  # xcorr: direction for shiftting along y axis
        img[inx] = cv2.warpAffine(img[inx], T, img.shape[3:0:-1])

    if parallel:
        _ = Parallel(n_jobs=n_jobs, require='sharedmem')(delayed(parallel_drift)(i) for i in range(img.shape[0]))
    else:
        for i in range(img.shape[0]):
            if drift is None:
                frame = rangescale(img[i], (0, 255)).astype(np.uint8)  # Making sure its the right format
                driftcorrimg = cropbox(frame, box)  # crop part of image for matching
                res = cv2.matchTemplate(driftcorrimg, template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)  # top left corner
                ycorr[i] = max_loc[0] - res.shape[1] / 2  # TODO: why size should over 2 ?
                xcorr[i] = max_loc[1] - res.shape[0] / 2  # TODO: why size should over 2 ?
            T = np.float32([[1, 0, -ycorr[i]],
                            [0, 1, -xcorr[i]]])  # xcorr: direction for shiftting along y axis
            img[i] = cv2.warpAffine(img[i], T, img.shape[3:0:-1])

    if twoDflag:
        return np.squeeze(img), (xcorr[0], ycorr[0])
    else:
        return img, (xcorr, ycorr)


def getChamberBoxes(chambersmask) -> list:
    '''
    This function extracts the bounding boxes of the chambers in thebinary mask
    produced by the chambers identification unet

    Parameters
    ----------
    chambersmask : 2D array of uint8/uint16/floats
        The mask of the chambers as returned by the chambers id unet.

    Returns
    -------
    chamberboxes : list
        List of cropping box dictionaries (see cropbox()).

    '''
    chamberboxes = []
    if chambersmask.dtype == bool:
        chambersmask = chambersmask.astype(np.uint8)
    else:
        chambersmask = cv2.threshold(chambersmask, .5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = cv2.findContours(chambersmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    y_length, _ = chambersmask.shape
    for chamber in contours:
        xtl, ytl, boxwidth, boxheight = cv2.boundingRect(chamber)
        chamber_dict = dict(
            xtl=xtl,
            ytl=int(ytl - (boxheight * .08)),  # -10% of height to make sure the top isn't cropped
            xbr=xtl + boxwidth,
            ybr=int(ytl + boxheight * 1.1))

        if chamber_dict['ytl'] < 0:
            chamber_dict['ytl'] = 0
        if chamber_dict['ybr'] > y_length:
            chamber_dict['ybr'] = y_length

        chamberboxes.append(chamber_dict)  # tl = top left, br = bottom right. MODIFY: add more 10% of height

    chamberboxes.sort(key=lambda elem: elem['xtl'])  # Sorting by top-left X (normally sorted by Y top left)
    return chamberboxes


def getBoundingRect(chambers_mask: np.ndarray) -> List[Dict[str, int]]:
    """
    This function extracts the bounding boxes of the outlets in the binary mask
    produced by the chambers identification unet.

    Parameters
    ----------
    chambers_mask : 2D array of uint8/uint16/floats
        The mask of the chambers as returned by the chambers id unet.

    Returns
    -------
    outboxes : List[Dict[str, int]]
        List of cropping box dictionaries (see cropbox()).
        it has 4 keys: xtl, ytl, xbr, ybr
        xtl : int
            the axis 1 coordinate of the top-left corner of the box.
        ytl : int
            the axis 0 coordinate of the top-left corner of the box.
        xbr : int
            the axis 1 coordinate of the bottom-right corner of the box.
        ybr : int
            the axis 0 coordinate of the bottom-right corner of the box.

    """
    outboxes = []
    chamber_num = int(chambers_mask.max())
    for chamber_value in range(1, chamber_num + 1):
        single_chamber_mask = (chambers_mask == chamber_value).astype(np.uint8)
        contours = cv2.findContours(single_chamber_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            xtl, ytl, boxwidth, boxheight = cv2.boundingRect(contour)
            outboxes.append(dict(xtl=xtl, ytl=ytl, xbr=xtl + boxwidth, ybr=ytl + boxheight))

    return outboxes


def getChamberBoxes_CPM(chambers_mask: np.ndarray, wide_channer=True, flip=False) -> List[Dict[str, int]]:
    """
    This function extracts the bounding boxes of the chambers in the binary mask
    produced by the chambers identification unet

    Parameters
    ----------
    flip : bool, optional
        this flag determines the orientation of the cropping box.
        It diciates the outlet of the channel is at the top of the image.
    chambers_mask : 2D array of uint8/uint16/floats
        The mask of the chambers as returned by the chambers id unet.
    wide_channer : bool, optional
        If True, the function will return wider cropping boxes to include more
        of the channel. Default is True.

    Returns
    -------
    chamberboxes : List[Dict[str, int]]
        List of cropping box dictionaries (see cropbox()).

    Change log
    ----------
    - 2025-6-4: Added wide_channer parameter to extend the box to include more of the channel.

    """

    chamberboxes = []

    chamber_num = int(chambers_mask.max())

    for chamber_value in range(1, chamber_num + 1):

        single_chamber_mask = (chambers_mask == chamber_value).astype(np.uint8)
        contour = cv2.findContours(single_chamber_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        y_length, x_length = single_chamber_mask.shape
        xtl, ytl, boxwidth, boxheight = cv2.boundingRect(contour)
        if flip is False:
            chamber_dict = dict(
                xtl=int(xtl),
                ytl=int(ytl - boxheight * .08),  # -8% of height to make sure the top isn't cropped
                xbr=int(xtl + boxwidth),
                ybr=int(ytl + boxheight * 0.95)
            )  # -5% of height to make sure the outlet of channel is cropped.
        else:
            chamber_dict = dict(
                xtl=int(xtl),
                ytl=int(ytl - boxheight * .05),  # +5% of height to make sure the outlet of channel is cropped.
                xbr=int(xtl + boxwidth),
                ybr=int(ytl + boxheight * 1.08)  # +8% of height to make sure the top isn't cropped
            )
        if chamber_dict['ytl'] < 0:
            chamber_dict['ytl'] = 0
        if chamber_dict['ybr'] > y_length:
            chamber_dict['ybr'] = y_length
        if wide_channer:
            # If wide channel, extend the box to include more of the channel
            chamber_dict['xtl'] = max(0, int(chamber_dict['xtl'] - boxwidth*0.5))
            chamber_dict['xbr'] = min(x_length, int(chamber_dict['xbr'] + boxwidth*0.5))
        chamber_dict['xtl'] = max(0, chamber_dict['xtl'])
        chamber_dict['xbr'] = min(x_length, chamber_dict['xbr'])
        chamberboxes.append(chamber_dict)
    chamberboxes.sort(key=lambda elem: elem['xtl'])  # Sorting by top-left X (normally sorted by Y top left)

    return chamberboxes


def getDriftTemplate(chamber_boxes: Iterable[Dict[str, int]],
                     img: np.ndarray) -> Union[np.ndarray, None]:
    """
    This function retrieves a region above the chambers to use as drift template

    Parameters
    ----------
    chamber_boxes : Iterable[Dict[str, int]]
        See getChamberBoxes().
    img : np.ndarray
        The first frame of a movie to use as reference for drift correction.

    Returns
    -------
    2D numpy array or None
        A cropped region of the image to use as template for drift correction.
        If an empty list of chamber boxes is passed, None is returned.
        (see driftcorr()).

    """

    if len(chamber_boxes) == 0:
        return None
    (y_cut, x_cut) = [round(i * .01) for i in img.shape]  # Cutting out 1% (1% origin) of the image on each side as drift margin
    ybr = max(chamber_boxes, key=lambda elem: elem['ytl'])['ytl'] - y_cut
    if ybr < y_cut:
        ybr=max(chamber_boxes, key=lambda elem: elem['ytl'])['ytl'] + y_cut
    box = dict(
        xtl=x_cut,
        xbr=-x_cut,
        ytl=y_cut,
        ybr=ybr
    )
    # if box['ybr'] < box['ytl']
    # ybr=max(chamberboxes, key=lambda elem: elem['ytl'])['ytl'] + y_cut)
    return cropbox(img, box)


def getSinglecells(seg):
    """
    Return masks of single cells

    Parameters
    ----------
    seg : array of uint8/uint16/float/bool
        Mask of cells. Values >0.5 will be considered cell pixels

    Returns
    -------
    singlecellseg : 3D array of uint8
        Stack of single cell masks. Each single-cell mask is stacked along the
        first axis (axis 0)

    """
    if seg.dtype == bool:
        seg = seg.astype(np.uint8)
    else:
        seg = cv2.threshold(seg, .5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = list(cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    contours.sort(key=lambda elem: np.max(elem[:, 0, 1]))  # Sorting along Y
    singlecellseg = np.empty([len(contours), seg.shape[0], seg.shape[1]])
    for c, contour in enumerate(contours):
        singlecellseg[c] = cv2.drawContours(np.zeros((seg.shape[0], seg.shape[1]), dtype=np.uint8), [contour], 0, color=1, thickness=-1)
    return singlecellseg


def getOverlap(output, target):
    """
    Get portion of tracking output overlapping on target cell

    Parameters
    ----------
    output : array of uint8
        Mask of tracking output.
    target : array of uint8
        Mask of target cell.

    Returns
    -------
    float
        Portion of target cell that is covered by the tracking output.

    """
    return cv2.sumElems(
        cv2.bitwise_and(output,
                        target)
    )[0] / \
        cv2.sumElems(
            target
        )[0]


def filter_small_cells(segmentation_mask: np.ndarray, min_size: int = 80) -> np.ndarray:
    """
    Filter out small cells from a segmentation mask.
    Cells are label with values >= 1, and background is 0.
    Cells smaller than min_size will be removed from the mask.
    Parameters
    ----------
    segmentation_mask: np.ndarray
        2D array of uint8/uint16/floats/bool representing the segmentation mask.
        Cells are labeled with values >= 1, and background is 0.
    min_size: int, optional
        Minimum size of cells to keep in the mask. Default is 100 pixels.
        Cells smaller than this size will be removed.

    Returns
    -------
    np.ndarray
        Filtered segmentation mask with small cells removed.
        Cells are labeled with values >= 1, and background is 0.
        Small cells are set to 0.

    """
    cells_label = np.unique(segmentation_mask)
    cells_label = cells_label[cells_label > 0]  # Exclude background (0)
    filtered_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    new_label = 1
    for label in cells_label:
        cell_mask = (segmentation_mask == label).astype(np.uint8)
        cell_size = np.sum(cell_mask)
        if cell_size >= min_size:
            filtered_mask += cell_mask * new_label
            new_label += 1
    return filtered_mask


def label_seg(seg, cellnumbers=None):
    '''
    Label cells in segmentation mask

    Parameters
    ----------
    seg : numpy 2D array of float/uint8/uint16/bool
        Cells segmentation mask. Values >0.5 will be considered cell pixels
    cellnumbers : list of ints, optional
        Numbers to attribute to each cell mask, from top to bottom of image.
        Because we are using uint16s, maximum cell number is 65535. If None is
        provided, the cells will be labeled 1,2,3,... Background is 0
        The default is None.

    Returns
    -------
    label : 2D numpy array of uint16
        Labelled image. Each cell in the image is marked by adjacent pixels
        with values given by cellnumbers

    '''
    if seg.dtype == bool:
        seg = seg.astype(np.uint8)
    else:
        seg = cv2.threshold(seg, .5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = list(cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    contours.sort(key=lambda elem: np.max(elem[:, 0, 1]))  # Sorting along Y
    label = np.zeros(seg.shape, dtype=np.uint16)  # Because we use uint16, we can not have >65535 cells in one movie
    for c, contour in enumerate(contours):
        label = cv2.fillPoly(label, [contour], c + 1 if cellnumbers is None else cellnumbers[c])
    return label


class xpreader:
    """Experiment reader for TIFF files and single-page TIFF sequence folders.

    .. note::
        Ported from Moma_cellPose/utilities.py into moma_pipeline. The
        bioformats/javabridge reader path was REMOVED in the port (it only
        ever worked on Python 2). The 'use_bioformats' parameter is kept in
        the signature for compatibility but passing True raises
        NotImplementedError. TIFF stacks are read through 'tifffile'
        (replacing the removed 'skimage.external.tifffile'); the
        image-sequence directory path is unchanged.

    """

    def __init__(self, filename, channelnames=None, use_bioformats=False, prototype=None, fileorder='pct',
                 filenamesindexing=1):
        """
        Initialize experiment reader

        Parameters
        ----------
        filename : String
            Path to experiment tif file or directory.If the path leads to a
            directory, the experiment folder is expected to contain exclusively
            single-page tif images. If no prototype is provided, the filenames
            are expected to be of the following C-style format:
            %s%d%s%d%s%d.tif, with the 3 %d digit strings being zero-padded
            decimal representation of the position, channel and frame/timepoint
            number of each image file.
            Valid examples:
                Pos01_Cha3_Fra0005.tif
                p3c1t034.tif
                xy 145 - fluo 029 - timepoint 005935 .TIFF
        channelnames : List/tuple of strings or None, optional
            Names of the acquisition channels ('trans', 'gfp', ...).
            The default is None.
        use_bioformats : bool, optional
            Flag to use the bioformats reader.
            REMOVED in the moma_pipeline port: the bioformats/javabridge path
            only worked on Python 2 and has been deleted. The parameter is kept
            for signature compatibility; passing True raises NotImplementedError.
            The default is False.
        prototype: string, optional
            Filename prototype to use when reading single-page tif images from
            a sequence folder, in C-style formatting. Folder separators can be
            used. If None, the prototype will be estimated from the first tif
            file in the folder. For example, an experiment from micromanager
            can be processed with prototype =
            'Pos%01d/img_channel%03d_position%03d_time%09d_z000.tif'
            and fileorder = 'pcpt' and filenamesindexing = 0
            The default is None.
        fileorder: string, optional
            Order of the numbers in the prototype, with 'p' for positions/
            series, 'c' for imaging channels, and 't' for timepoints/frames.
            For example 'pct' indicates that the first number is going to be
            positions, then channels, then timepoints. You can use the same
            letter multiple times, like 'pcpt'.
            The default is 'pct'
        filenamesindexing = int
            Selects between 0-based or 1-based indexing in the filename. If
            1, position 0 will be referenced as position 1 in the filename.
            The default is 1


        Raises
        ------
        ValueError
            If the filenames in the experimental directory do not follow the
            correct format, a ValueError will be raised.

        Returns
        -------
        None.

        """

        _, file_extension = os.path.splitext(filename)
        if use_bioformats:
            # Removed in the moma_pipeline port: the bioformats/javabridge
            # reader only worked on Python 2.
            raise NotImplementedError(
                'The bioformats/javabridge reader was removed in the moma_pipeline '
                'port; use_bioformats is no longer supported. Use the tifffile-based '
                'TIFF reader (default) instead.')
        self.channelnames = channelnames
        if os.path.isdir(filename):  # Experiment is stored as individual image TIFF files in a folder
            self.filetype = 'dir'
            self.filehandle = filename
            self.fileorder = fileorder
            self.filenamesindexing = filenamesindexing
            # If filename prototype is not provided, guess it from the first file:
            if prototype is None:
                imgfiles = [x for x in os.listdir(filename) if os.path.splitext(x)[1].lower() in ('.tif', '.tiff')]
                # Here we assume all images in the folder follow the same naming convention:
                numstrs = re.findall("\d+", imgfiles[0])  # Get digits sequences in first filename
                charstrs = re.findall("\D+", imgfiles[0])  # Get character sequences in first filename
                if len(numstrs) != 3 or len(charstrs) != 4:
                    raise ValueError('Filename formatting error. See documentation for image sequence formatting')
                # Create the string prototype to be used to generate filenames on the fly:
                # Order is position, channel, frame/timepoint
                self.prototype = charstrs[0] + '%0' + str(len(numstrs[0])) + 'd' + \
                                 charstrs[1] + '%0' + str(len(numstrs[1])) + 'd' + \
                                 charstrs[2] + '%0' + str(len(numstrs[2])) + 'd' + \
                                 charstrs[3]
            else:
                self.prototype = prototype
            # Get experiment settings by testing if relevant files exist:
            # Get number of positions:
            self.positions = 0
            while (os.path.exists(self.getfilenamefromprototype(self.positions, 0, 0))): self.positions += 1
            # Get number of channels:
            self.channels = 0
            while (os.path.exists(self.getfilenamefromprototype(0, self.channels, 0))): self.channels += 1
            # Get number of frames/timepoints:
            self.timepoints = 0
            while (os.path.exists(self.getfilenamefromprototype(0, 0, self.timepoints))): self.timepoints += 1
            # Get image specs:
            I = cv2.imread(self.getfilenamefromprototype(0, 0, 0), cv2.IMREAD_ANYDEPTH)
            self.x = I.shape[1]
            self.y = I.shape[0]
            self.dtype = I.dtype

        elif (
                file_extension.lower() == '.tif' or file_extension.lower() == '.tiff'):  # Works with single-series tif & mutli-series ome.tif
            self.filetype = 'tif'
            self.filehandle = tiff.TiffFile(filename)
            self.positions = len(self.filehandle.series)
            s = self.filehandle.series[0]  # Here I'm going to assume all series have the same format
            self.timepoints = s.shape[s.axes.find('T')]
            self.channels = s.shape[s.axes.find('C')]
            self.x = s.shape[s.axes.find('X')]
            self.y = s.shape[s.axes.find('Y')]
            self.dtype = s.pages[0].asarray().dtype

    def close(self):
        if self.filetype == 'tif':  # Nothing to do if sequence directory
            self.filehandle.close()

    def getfilenamefromprototype(self, position, channel, frame):
        '''
        Generate full filename for specific frame based on file path,
        prototype, fileorder, and filenamesindexing

        Parameters
        ----------
        position : int
            Position/series index (0-based indexing).
        channel : int
            Imaging channel index (0-based indexing).
        frame : int
            Frame/timepoint index (0-based indexing).

        Returns
        -------
        string
            Filename.

        '''
        filenumbers = []

        for i in self.fileorder:
            if i == 'p':
                filenumbers.append(position + self.filenamesindexing)
            if i == 'c':
                filenumbers.append(channel + self.filenamesindexing)
            if i == 't':
                filenumbers.append(frame + self.filenamesindexing)
        return os.path.join(self.filehandle, self.prototype % tuple(filenumbers))

    def getframes(self, positions=None, channels=None, frames=None,
                  squeeze_dimensions=True,
                  resize=None,
                  rescale=None,
                  rotate=None):
        """
        Get frames from experiment.

        Parameters
        ----------
        positions : None, int, tuple/list of ints, optional
            The frames from the position index or indexes passed as an integer
            or a tuple/list will be returned. If None is passed, all positions
            are returned.
            The default is None.
        channels : None, int, tuple/list of ints, str, tuple/list of str, optional
            The frames from the channel index or indexes passed as an integer
            or a tuple/list will be returned. If the channel names have been
            defined, the channel(s) can be passed as a string or tuple/list of
            strings. If None is passed, all channels are returned.
            The default is None.
        frames : None, int, tuple/list of ints, optional
            The frame index or indexes passed as an integer or a tuple/list
            will be returned. If None is passed, all frames are returned.
            The default is None.
        squeeze_dimensions : bool, optional
            If True, the numpy squeeze function is applied to the output array,
            removing all singleton dimensions.
            The default is True.
        resize : None or tuple/list of 2 ints, optional
            Dimensions to resize the frames. If None, no resizing is performed.
            The default is None.
        rescale : None or tuple/list of 2 int/floats, optional
            Rescale all values in each frame to be within the given range.
            The default is None.
        rotate : None or float, optional
            Rotation to apply to the image (in degrees).
            The default is None.

        Raises
        ------
        ValueError
            If channel names are not correct.

        Returns
        -------
        Numpy Array
            Concatenated frames as requested by the different input options.
            If squeeze_dimensions=False, the array is 5-dimensional, with the
            dimensions order being: Position, Time, Channel, Y, X

        """

        # Handle options:
        if frames is None:  # No frames specfied: load all
            frames = list(range(self.timepoints))
        elif type(frames) is not list and type(frames) is not tuple:
            frames = [frames]

        if positions is None:  # No positions specified: load all
            positions = list(range(self.positions))
        elif type(positions) is not list and type(positions) is not tuple:
            positions = [positions]

        if channels is None:  # No channel specified: load all
            channels = list(range(self.channels))
        elif type(channels) is str:  # refer to 1 channel by name (eg 'gfp')
            if self.channelnames is None:
                raise ValueError('Set channel names first')
            if channels in self.channelnames:
                channels = [self.channelnames.index(channels)]
            else:
                raise ValueError(channels + ' is not a valid channel name.')
        elif (type(channels) is list or type(channels) is tuple) and all(
                [type(c) is str for c in channels]):  # refer to channels by list/tuple of names:
            if self.channelnames is None:
                raise ValueError('Set channel names first')
            for i, c in enumerate(channels):
                if c in self.channelnames:
                    channels[i] = self.channelnames.index(c)
                else:
                    raise ValueError(c + ' is not a valid channel name.')
        elif type(channels) is not list and type(channels) is not tuple:
            channels = [channels]

        # Allocate memory:
        if rescale is None:
            dt = self.dtype
        else:
            dt = np.float32
        if resize is None:
            output = np.empty([len(positions), len(frames), len(channels), self.y, self.x], dtype=dt)
        else:
            output = np.empty([len(positions), len(frames), len(channels), resize[0], resize[1]], dtype=dt)

        # Load images:
        for p, pos in enumerate(positions):
            for c, cha in enumerate(channels):
                for f, fra in enumerate(frames):
                    # Read frame:
                    if self.filetype == 'dir':
                        frame = cv2.imread(self.getfilenamefromprototype(pos, cha, fra), cv2.IMREAD_ANYDEPTH)
                    elif self.filetype == 'tif':
                        frame = self.filehandle.series[pos]. \
                            pages[fra * self.channels + cha]. \
                            asarray()
                    # Optionally resize and rescale:
                    if rotate is not None:
                        M = cv2.getRotationMatrix2D((self.x / 2, self.y / 2), rotate, 1)
                        frame = cv2.warpAffine(frame, M, (self.x, self.y), borderMode=cv2.BORDER_REPLICATE)
                    if resize is not None:
                        frame = cv2.resize(frame, resize[::-1])  # cv2 inverts shape
                    if rescale is not None:
                        frame = rangescale(frame, rescale)
                    # Add to output array:
                    output[p, f, c, :, :] = frame

        # Return:
        return np.squeeze(output) if squeeze_dimensions else output
