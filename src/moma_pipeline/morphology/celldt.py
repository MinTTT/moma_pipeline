"""
CellDT: distance-transform cell morphology.

Ported from Moma_cellPose/cell_morphology_compat.py into the
moma_pipeline.morphology package (voronoi core: moma_pipeline.morphology.voronoi_dt).

Provides a drop-in replacement for the Cell class used in
Moma_CPM_FOV.py.  Uses dt_medial_axis + extract_morphology
internally while exposing the same attribute names and
export/load interface as cell_coord_utility.Cell.

Usage in Moma_CPM_FOV.py (future):
    from moma_pipeline.morphology.celldt import CellDT
    cell = CellDT(cell_index=cell_i, fov=time_key, umppx=0.065)
    cell.compute_from_mask(cell_mask, cells_images)
    # cell now has .spine_length, .area, .width, .flu_level, etc.

Or process a whole channel dict:
    from moma_pipeline.morphology.celldt import process_channel_masks
    cells = process_channel_masks(cell_masks_dict, channel_images_dict,
                                   time_points, umppx=0.065)
"""

import numpy as np
from numba import njit
from .voronoi_dt import (
    DEFAULT_STEP_SIZE, DEFAULT_SMOOTH_TOL, dt_medial_axis, extract_morphology,
)

DEFAULT_UM_PER_PX = 0.065


class CellDT:
    """Cell-like object backed by the distance-transform morphology algorithm.

    Exposes the same attribute interface as cell_coord_utility.Cell:
        cell_index, fov, umppx
        spine_length, area, width, width_area
        x_min, x_max, y_min, y_max
        flu_level (dict: {channel: {mean, median, total}})
        spine, skeleton, centerline
        cellSkeletonSuccess

    Also provides export_cell_parameters / load_cell_parameters for
    pickle-based serialization compatibility.
    """

    def __init__(self, cell_index=None, fov=None, umppx=DEFAULT_UM_PER_PX):
        self.cell_index = cell_index
        self.fov = fov
        self.umppx = umppx

        # Morphology (in pixel units until converted)
        self._length_px = 0.0
        self._spine_length_px = 0.0
        self._area_px = 0.0
        self._mask_area_px = 0.0
        self._width_px = 0.0
        self._max_width_px = 0.0
        self._volume_px = 0.0
        self._curvature = 0.0
        self._n_ribs = 0
        self._pole_radius = 0.0
        self._endpoint_radii = (0.0, 0.0)
        self.geometry_mode = 'invalid'
        self.spine_mode = 'default'
        self.channel_axis = (0.0, 1.0)
        self.pole_refinement = 'none'
        self.pole_refinement_diagnostics = {}

        # In um (matching Cell attribute names)
        self.spine_length = 0.0
        self.area = 0.0
        self.mask_area = 0.0
        self.width = 0.0
        self.width_area = 0.0

        # Bounding box
        self.x_min = 0
        self.x_max = 0
        self.y_min = 0
        self.y_max = 0

        # For compatibility
        self.spine = None       # np.ndarray (N, 2) centerline in [x, y]
        self.skeleton = None    # np.ndarray raw skeleton points
        self.centerline = None  # np.ndarray (M, 2) interpolated centerline
        self.mesh = None        # np.ndarray (M, 4) raw mesh
        self.cellSkeletonSuccess = False

        # Fluorescence
        self.channel_imgs = {}
        self.background_fluorescence = {}
        self.flu_level = {}
        self.flu_distribution = {}
        self._mask_indices = (np.array([], dtype=np.intp),
                              np.array([], dtype=np.intp))
        self._mask_shape = None

        # Other Cell attributes for serialization compatibility
        self.channels = []
        self.fov_size = None
        self.fov_size_cv2 = None
        self.xLen = None
        self.yLen = None
        self.contour_init = None
        self._contour_opt = None
        self.edge_mask = None
        self.edge_mask_xy = None
        self.skeleton_func = None
        self.rectangle = None
        self.verticalCell = None
        self.snapSlice = []
        self.cellCoord = False

    # ------------------------------------------------------------------
    # Main compute method
    # ------------------------------------------------------------------

    def compute_from_mask(self, cell_mask, channel_images=None,
                          step_size=DEFAULT_STEP_SIZE,
                          smooth_tol=DEFAULT_SMOOTH_TOL,
                          flu_distribution_channels=None,
                          flu_distribution_bins=15,
                          background_fluorescence=None,
                          spine_mode="default", channel_axis=(0.0, 1.0),
                          pole_refinement="none"):
        """Compute morphology from a binary cell mask.

        Parameters
        ----------
        cell_mask : np.ndarray, 2D bool
            Binary mask of a single cell.
        channel_images : dict or None
            {channel_name: np.ndarray} for fluorescence calculation.
        step_size : float or None
        smooth_tol : float
        flu_distribution_channels : list or None
            Channels for fluorescence distribution calculation. None disables
            distribution calculation at this level.
        flu_distribution_bins : int
            Number of relative bins from old pole (-1) to new pole (1).
            The default is 15; 31 is retained only for legacy reproduction.
        background_fluorescence : dict or None
            Per-channel background values. Each value can be a scalar or a
            dictionary containing ``mean``. The background mean is used for
            the corrected total fluorescence calculation.
        pole_refinement : {"none", "contour"}
            Compatibility choice. For channel_axis, one contour body refinement
            attempt runs at most 15 damped iterations (beta=0.45), stopping when
            mean matched spine shift is <0.25 px and retaining the best valid
            coverage-preserving body at the limit. The initial complete contour
            reference is saved before updates; seed endpoints and initial radii
            remain fixed (shared rods, local independent branch radii). Pole
            centers fit inward within saved bounds against fixed local contour
            arcs. Final length and tip coverage must pass; failure restores the
            exact initial reference. Terminal cleanup precedes morphology. "none" is
            deprecated in this mode.
        """
        if pole_refinement not in {"none", "contour"}:
            raise ValueError("pole_refinement must be none or contour")
        if pole_refinement == "contour" and spine_mode != "channel_axis":
            raise ValueError("contour pole_refinement requires spine_mode=channel_axis")
        self.pole_refinement = "contour" if spine_mode == "channel_axis" else pole_refinement
        self.pole_refinement_diagnostics = {}
        if spine_mode == "channel_axis":
            mesh, ep_radii, geometry_mode, diagnostics = dt_medial_axis(
                cell_mask, step_size=step_size, smooth_tol=smooth_tol,
                return_geometry=True, spine_mode=spine_mode, channel_axis=channel_axis,
                pole_refinement="contour", return_diagnostics=True)
            self.pole_refinement_diagnostics = diagnostics
        else:
            # Preserve the baseline call signature for existing adapters/mocks.
            mesh, ep_radii, geometry_mode = dt_medial_axis(
                cell_mask, step_size=step_size, smooth_tol=smooth_tol,
                return_geometry=True, spine_mode=spine_mode, channel_axis=channel_axis)
        self.geometry_mode = str(geometry_mode)
        self.spine_mode = spine_mode
        self.channel_axis = tuple(float(v) for v in channel_axis)
        if mesh is None:
            self._clear_failed_morphology()
            return

        morph = extract_morphology(mesh, endpoint_radii=ep_radii)
        self._set_from_morphology(
            morph, mesh, ep_radii, geometry_mode, cell_mask, channel_images,
            flu_distribution_channels=flu_distribution_channels,
            flu_distribution_bins=flu_distribution_bins,
            background_fluorescence=background_fluorescence,
        )

    def _clear_failed_morphology(self):
        """Discard previous results while retaining identity, inputs and request."""
        defaults = CellDT()
        for name in (
                '_length_px', '_spine_length_px', '_area_px', '_mask_area_px',
                '_width_px', '_max_width_px', '_volume_px', '_curvature',
                '_n_ribs', '_pole_radius', '_endpoint_radii', 'geometry_mode',
                'spine_length', 'area', 'mask_area', 'width', 'width_area',
                'x_min', 'x_max', 'y_min', 'y_max', 'spine', 'skeleton',
                'centerline', 'mesh', 'cellSkeletonSuccess', 'flu_level',
                'flu_distribution', '_mask_indices', '_mask_shape',
                'contour_init', '_contour_opt', 'edge_mask', 'edge_mask_xy',
                'skeleton_func', 'rectangle', 'verticalCell', 'snapSlice', 'cellCoord'):
            setattr(self, name, getattr(defaults, name))
        self._centerline = None
        self._steplength = np.array([])
        self._lengthvector = np.array([])

        # load_cell_parameters accepts historical public aliases as well as
        # current fields. Clear any such aliases without discarding raw images,
        # background values, custom metadata, or the rejection diagnostics.
        for name in ('curvature', 'max_width_px', 'n_ribs', 'pole_radius',
                     'volume_px', 'body_spine_length_px', 'volume_um3',
                     'max_width_um', 'pole_radius_um'):
            if hasattr(self, name):
                setattr(self, name, 0.0)
        if hasattr(self, 'endpoint_radii'):
            self.endpoint_radii = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Internal: populate attributes from morphology dict
    # ------------------------------------------------------------------

    def _set_from_morphology(self, morph, mesh, endpoint_radii, geometry_mode,
                             cell_mask, channel_images,
                             flu_distribution_channels=None,
                             flu_distribution_bins=15,
                             background_fluorescence=None):
        """Fill Cell attributes from morphology dict and mask."""
        # Pixel-level values
        self._length_px = morph.get('length', 0.0)
        self._spine_length_px = morph.get('spine_length', 0.0)
        self._area_px = morph.get('area', 0.0)
        self._mask_area_px = float(np.count_nonzero(cell_mask))
        self._mask_indices = np.nonzero(cell_mask)
        self._mask_shape = tuple(cell_mask.shape)
        self._width_px = morph.get('mean_width', 0.0)
        self._max_width_px = morph.get('max_width', 0.0)
        self._volume_px = morph.get('volume', 0.0)
        self._curvature = morph.get('curvature', 0.0)
        self._n_ribs = morph.get('n_ribs', 0)
        self._pole_radius = morph.get('pole_radius', 0.0)
        self._endpoint_radii = tuple(float(r) for r in endpoint_radii)
        self.geometry_mode = str(geometry_mode)
        self._steplength = morph.get('steplength', np.array([]))
        self._lengthvector = morph.get('lengthvector', np.array([]))

        # Convert to um (matching original Cell behavior)
        um2 = self.umppx ** 2
        um3 = self.umppx ** 3
        self.spine_length = self._length_px * self.umppx
        self.area = self._area_px * um2
        self.mask_area = self._mask_area_px * um2
        self.width = self._width_px * self.umppx
        self.width_area = self._width_px * self.umppx  # same as width for DT method

        # Bounding box from mask
        ys, xs = np.where(cell_mask)
        self.x_min = int(xs.min())
        self.x_max = int(xs.max())
        self.y_min = int(ys.min())
        self.y_max = int(ys.max())

        # Spine / centerline
        self.centerline = morph.get('centerline', None)
        self.spine = self.centerline  # compatibility
        self.skeleton = self.centerline
        self.mesh = mesh

        # Fluorescence
        if channel_images is not None:
            self.channel_imgs = channel_images
            self.background_fluorescence = background_fluorescence or {}
            self._compute_fluorescence()
            if flu_distribution_channels is not None:
                self.compute_flu_distribution(
                    channels=flu_distribution_channels,
                    n_bins=flu_distribution_bins,
                    cell_mask=cell_mask,
                )

        self.cellSkeletonSuccess = True

    # ------------------------------------------------------------------
    # Fluorescence (compatible with original Cell.cal_cel_flu_level)
    # ------------------------------------------------------------------

    def _compute_fluorescence(self, channels=None,
                              background_fluorescence=None):
        """Compute fluorescence from the unique pixels in the cell mask.

        ``total`` is background corrected using the mask pixel count as the
        cell area in pixels::

            total = sum(mask_pixel_values) - background_mean * mask_pixel_count

        The uncorrected value, applied background, and pixel count are retained
        alongside ``total`` so the calculation remains auditable.
        """
        if background_fluorescence is not None:
            self.background_fluorescence = background_fluorescence

        if channels is None:
            channels = list(self.channel_imgs.keys())

        self.flu_level = {}
        for ch_name in channels:
            if ch_name not in self.channel_imgs:
                continue
            img = self.channel_imgs[ch_name]
            flu_vals = self._sample_from_mask(img)
            background = self._get_background_mean(ch_name)
            pixel_count = int(len(flu_vals))
            if len(flu_vals) > 0:
                raw_total = float(np.sum(flu_vals, dtype=np.float64))
                raw_mean = float(np.mean(flu_vals))
                self.flu_level[ch_name] = {
                    'mean': raw_mean,
                    'raw_mean': raw_mean,
                    'background_corrected_mean': raw_mean - background,
                    'median': float(np.median(flu_vals)),
                    'total': raw_total - background * pixel_count,
                    'raw_total': raw_total,
                    'background': background,
                    'pixel_count': pixel_count,
                }
            else:
                self.flu_level[ch_name] = {
                    'mean': 0.0, 'median': 0.0, 'total': 0.0,
                    'raw_mean': 0.0, 'background_corrected_mean': np.nan,
                    'raw_total': 0.0, 'background': background,
                    'pixel_count': 0,
                }

    def _sample_from_mask(self, img):
        """Return each image pixel selected by the cell mask exactly once."""
        rows, cols = self._mask_indices
        if len(rows) == 0:
            return np.array([])
        if img.ndim != 2:
            raise ValueError('fluorescence images must be two-dimensional')
        if self._mask_shape is not None and tuple(img.shape) != self._mask_shape:
            raise ValueError('cell mask and fluorescence image shapes do not match')
        if rows[-1] >= img.shape[0] or cols.max() >= img.shape[1]:
            raise ValueError('cell mask and fluorescence image shapes do not match')
        return img[rows, cols].astype(np.float64, copy=False)

    def _get_background_mean(self, channel):
        """Resolve one channel's background mean, defaulting to zero."""
        background = self.background_fluorescence.get(channel, 0.0)
        if isinstance(background, dict):
            background = background.get('mean', 0.0)
        if background is None:
            return 0.0
        background = float(background)
        if not np.isfinite(background):
            raise ValueError(f'non-finite background for channel {channel!r}')
        return background

    def compute_flu_distribution(self, channels=None, n_bins=15, cell_mask=None):
        """Compute raw fluorescence distribution along the DT profile axis.

        Rods and short capsules use old pole (-1), center (0), new pole (1).
        Round cells use one deterministic mesh rib ordered by increasing image
        y, with image x as a tie breaker. Round endpoints have no biological
        polarity.
        Results are stored in ``self.flu_distribution`` with keys:
        median_<channel>, total_<channel>, pixel_count, abs_pos, rel_pos,
        plus background and background-corrected per-bin totals.  Retaining
        ``pixel_count`` is essential because an integrated local fluorescence
        signal cannot be background corrected after export without the area of
        the corresponding longitudinal bins.
        """
        if self.centerline is None or len(self.centerline) < 2:
            self.flu_distribution = {}
            return self.flu_distribution
        if cell_mask is None:
            self.flu_distribution = {}
            return self.flu_distribution
        if channels is None:
            channels = list(self.channel_imgs.keys())
        if isinstance(channels, str):
            channels = [channels]
        channels = [channel for channel in channels if channel in self.channel_imgs]
        if not channels:
            self.flu_distribution = {}
            return self.flu_distribution

        rows, cols = np.nonzero(cell_mask)
        if len(rows) == 0:
            self.flu_distribution = {}
            return self.flu_distribution

        profile_line, profile_length_px, profile_metadata = (
            self._fluorescence_profile_line())
        points_xy = np.column_stack([cols.astype(np.float64), rows.astype(np.float64)])
        abs_pos_px = _project_points_to_polyline(points_xy, profile_line)
        rel_pos_pixels = np.clip(2.0 * abs_pos_px / profile_length_px - 1.0, -1.0, 1.0)

        n_bins = max(1, int(n_bins))
        rel_edges = np.linspace(-1.0, 1.0, n_bins + 1)
        rel_centers = (rel_edges[:-1] + rel_edges[1:]) / 2.0
        abs_pos_um = (rel_centers + 1.0) / 2.0 * profile_length_px * self.umppx
        bin_index = np.searchsorted(rel_edges, rel_pos_pixels, side='right') - 1
        bin_index = np.clip(bin_index, 0, n_bins - 1)

        distribution = {
            'abs_pos': abs_pos_um.astype(np.float64),
            'rel_pos': rel_centers.astype(np.float64),
            'pixel_count': np.bincount(
                bin_index, minlength=n_bins
            ).astype(np.int64),
            **profile_metadata,
        }
        for channel in channels:
            img = self.channel_imgs[channel]
            values = img[rows, cols].astype(np.float64, copy=False)
            (median_values, total_values, sum_squared_values,
             sample_variance_values) = _binned_fluorescence_statistics(
                values, bin_index, n_bins)
            background_mean = self._get_background_mean(channel)
            background_total = background_mean * distribution['pixel_count']
            background_corrected_total = total_values - background_total
            background_corrected_mean = np.divide(
                background_corrected_total,
                distribution['pixel_count'],
                out=np.full(n_bins, np.nan, dtype=np.float64),
                where=distribution['pixel_count'] > 0,
            )
            distribution[f'median_{channel}'] = median_values
            distribution[f'total_{channel}'] = total_values
            distribution[f'total_{channel}_raw'] = total_values
            distribution[f'sum_squared_{channel}_raw'] = sum_squared_values
            distribution[f'sample_variance_{channel}_raw'] = (
                sample_variance_values
            )
            distribution[f'sample_variance_{channel}_bg_corrected'] = (
                sample_variance_values.copy()
            )
            distribution[f'sem_{channel}_bg_corrected'] = np.sqrt(
                np.divide(
                    sample_variance_values,
                    distribution['pixel_count'],
                    out=np.full(n_bins, np.nan, dtype=np.float64),
                    where=distribution['pixel_count'] > 0,
                )
            )
            distribution[f'background_mean_{channel}'] = float(background_mean)
            distribution[f'background_total_{channel}'] = background_total.astype(
                np.float64
            )
            distribution[f'total_{channel}_background'] = background_total.astype(
                np.float64
            )
            distribution[f'background_corrected_total_{channel}'] = (
                background_corrected_total.astype(np.float64)
            )
            distribution[f'total_{channel}_bg_corrected'] = (
                background_corrected_total.astype(np.float64)
            )
            distribution[f'background_corrected_mean_{channel}'] = (
                background_corrected_mean
            )
            distribution[f'mean_{channel}_bg_corrected'] = (
                background_corrected_mean
            )
            distribution[f'background_corrected_median_{channel}'] = (
                median_values - background_mean
            )

        self.flu_distribution = distribution
        return self.flu_distribution

    def _fluorescence_profile_line(self):
        """Return the geometry-specific fluorescence profile polyline."""
        if self.geometry_mode == 'round':
            profile_line, profile_length_px = self._round_rib_profile_line()
            metadata = {
                'geometry_mode': 'round',
                'profile_axis': 'selected_rib',
                'profile_orientation': 'image_y_then_x_increasing',
                'axis_has_biological_polarity': False,
            }
            return profile_line, profile_length_px, metadata

        profile_line, profile_length_px = self._old_to_new_pole_line(
            preserve_order=self.geometry_mode == "channel_axis")
        metadata = {
            'geometry_mode': self.geometry_mode,
            'profile_axis': 'centerline',
            'profile_orientation': ('channel_axis_path_order'
                                   if self.geometry_mode == 'channel_axis' else
                                   'longest_body_path_order'
                                   if self.geometry_mode == 'branched_primary'
                                   else 'old_to_new_pole'),
            'axis_has_biological_polarity': self.geometry_mode not in {'branched_primary', 'channel_axis'},
        }
        return profile_line, profile_length_px, metadata

    def _round_rib_profile_line(self):
        """Return one deterministic round-cell rib as a diameter profile."""
        mesh = np.asarray(self.mesh, dtype=np.float64)
        if mesh.ndim != 2 or mesh.shape[0] < 1 or mesh.shape[1] < 4:
            raise ValueError('invalid_round_cell_rib')

        endpoint_a = mesh[0, 0:2].copy()
        endpoint_b = mesh[0, 2:4].copy()
        if (np.any(~np.isfinite(endpoint_a)) or
                np.any(~np.isfinite(endpoint_b))):
            raise ValueError('invalid_round_cell_rib')

        # Stable sign convention: increasing image y, then increasing image x.
        if ((endpoint_a[1], endpoint_a[0]) >
                (endpoint_b[1], endpoint_b[0])):
            endpoint_a, endpoint_b = endpoint_b, endpoint_a

        center = (endpoint_a + endpoint_b) / 2.0
        profile_line = np.vstack([endpoint_a, center, endpoint_b])
        profile_length_px = _polyline_length(profile_line)
        if profile_length_px <= 0 or not np.isfinite(profile_length_px):
            raise ValueError('invalid_round_cell_rib')
        return profile_line, profile_length_px

    def _old_to_new_pole_line(self, preserve_order=False):
        """Extend the ordered body; legacy rods use the image-y pole convention."""
        centerline = np.asarray(self.centerline, dtype=np.float64)
        if preserve_order or centerline[0, 1] <= centerline[-1, 1]:
            cl = centerline.copy()
            old_radius, new_radius = self._endpoint_radii
        else:
            cl = centerline[::-1].copy()
            old_radius = self._endpoint_radii[1]
            new_radius = self._endpoint_radii[0]

        old_tangent = _unit_vector(cl[1] - cl[0])
        new_tangent = _unit_vector(cl[-1] - cl[-2])
        if old_tangent is None or new_tangent is None:
            raise ValueError('invalid_centerline_tangent')

        old_tip = cl[0] - old_tangent * old_radius
        new_tip = cl[-1] + new_tangent * new_radius
        pole_line = np.vstack([old_tip, cl, new_tip])
        profile_length_px = _polyline_length(pole_line)
        if profile_length_px <= 0 or not np.isfinite(profile_length_px):
            raise ValueError('invalid_profile_length')
        return pole_line, profile_length_px

    # Compatibility wrappers that take channel_images separately
    def reassign_channel_imgs(self, channel_images):
        self.channel_imgs = channel_images

    def cal_cel_flu_level(self, channels=None, background_fluorescence=None):
        self._compute_fluorescence(
            channels=channels,
            background_fluorescence=background_fluorescence,
        )

    # ------------------------------------------------------------------
    # Serialization (matching Cell.export_cell_parameters / load_cell_parameters)
    # ------------------------------------------------------------------

    def export_cell_parameters(self):
        """Export cell parameters as a dict for pickle serialization."""
        return dict(
            cell_index=self.cell_index,
            fov=self.fov,
            umppx=self.umppx,
            spine_length=self.spine_length,
            area=self.area,
            mask_area=self.mask_area,
            width=self.width,
            width_area=self.width_area,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            flu_level=self.flu_level,
            curvature=self._curvature,
            max_width_px=self._max_width_px,
            n_ribs=self._n_ribs,
            pole_radius=self._pole_radius,
            endpoint_radii=self._endpoint_radii,
            geometry_mode=self.geometry_mode,
            spine_mode=self.spine_mode,
            channel_axis=list(self.channel_axis),
            pole_refinement=self.pole_refinement,
            pole_refinement_diagnostics=self.pole_refinement_diagnostics,
            cellSkeletonSuccess=self.cellSkeletonSuccess,
            flu_distribution=self.flu_distribution,
            # mesh geometry (for later reloading)
            mesh=self.mesh.tolist() if self.mesh is not None else None,
            _length_px=self._length_px,
            _spine_length_px=self._spine_length_px,
            _volume_px=self._volume_px,
            _area_px=self._area_px,
            _mask_area_px=self._mask_area_px,
        )

    def load_cell_parameters(self, params):
        """Load serialized scalars and restore the mesh-backed geometry."""
        for k, v in params.items():
            if k == 'mesh' and v is not None:
                self.mesh = np.array(v)
            elif k == 'centerline' and v is not None:
                self.centerline = np.array(v)
                self.spine = self.centerline
            else:
                setattr(self, k, v)

        # Exported public names historically differed from the internal names
        # read by the CSV exporter. Retain both, with explicit private values
        # taking precedence when a caller supplies them.
        for public, private in (('curvature', '_curvature'),
                                ('max_width_px', '_max_width_px'),
                                ('n_ribs', '_n_ribs'),
                                ('pole_radius', '_pole_radius')):
            if public in params and private not in params:
                setattr(self, private, params[public])
            # These are derived aliases, not independent measurements: keep
            # their pixel units and synchronize them with restored values.
            setattr(self, public, getattr(self, private))

        # Production pickles store the mesh but not a separate centerline.
        # Its rib midpoints are the same pixel coordinates used at compute time.
        if params.get('centerline') is None and self.mesh is not None:
            mesh = np.asarray(self.mesh)
            if mesh.ndim == 2 and mesh.shape[0] > 0 and mesh.shape[1] >= 4:
                self.centerline = (mesh[:, :2] + mesh[:, 2:4]) / 2.0
        if self.centerline is not None:
            self.spine = self.centerline
            self.skeleton = self.centerline

        if '_mask_area_px' not in params:
            if 'mask_area' in params and getattr(self, 'umppx', 0.0) > 0:
                self._mask_area_px = self.mask_area / (self.umppx ** 2)
            else:
                self._mask_area_px = 0.0
        if 'mask_area' not in params:
            self.mask_area = self._mask_area_px * (self.umppx ** 2)
        if 'endpoint_radii' in params:
            self._endpoint_radii = tuple(float(r) for r in params['endpoint_radii'])
        elif not hasattr(self, '_endpoint_radii'):
            self._endpoint_radii = (0.0, 0.0)
        if 'flu_distribution' not in params:
            self.flu_distribution = {}
        if 'geometry_mode' not in params:
            self.geometry_mode = _infer_geometry_mode_from_mesh(self.mesh)
        self.spine_mode = params.get('spine_mode',
                                     'channel_axis' if self.geometry_mode == 'channel_axis' else 'default')
        self.channel_axis = tuple(float(v) for v in params.get('channel_axis', (0.0, 1.0)))
        self.pole_refinement = params.get('pole_refinement', 'none')
        self.pole_refinement_diagnostics = params.get('pole_refinement_diagnostics', {})
        self.cellSkeletonSuccess = params.get('cellSkeletonSuccess', False)


@njit(nogil=True, cache=True)
def _unit_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.sqrt(np.sum(vector * vector)))
    if norm <= 1e-12 or not np.isfinite(norm):
        return None
    return vector / norm


def _infer_geometry_mode_from_mesh(mesh, tol=1e-12):
    """Infer round versus axis-bearing geometry for legacy serialized cells."""
    if mesh is None:
        return 'invalid'
    mesh = np.asarray(mesh, dtype=np.float64)
    if mesh.ndim != 2 or mesh.shape[0] < 2 or mesh.shape[1] < 4:
        return 'invalid'
    geometry_code = _infer_geometry_code_from_mesh(mesh, tol)
    if geometry_code < 0:
        return 'invalid'
    if geometry_code == 0:
        return 'round'
    return 'rod'


@njit(nogil=True, cache=True)
def _infer_geometry_code_from_mesh(mesh, tol):
    """Return -1 invalid, 0 round, or 1 axis-bearing for a mesh."""
    x0 = (mesh[0, 0] + mesh[0, 2]) / 2.0
    y0 = (mesh[0, 1] + mesh[0, 3]) / 2.0
    if not (np.isfinite(x0) and np.isfinite(y0)):
        return -1
    has_axis = False
    for i in range(1, len(mesh)):
        x = (mesh[i, 0] + mesh[i, 2]) / 2.0
        y = (mesh[i, 1] + mesh[i, 3]) / 2.0
        if not (np.isfinite(x) and np.isfinite(y)):
            return -1
        dx = x - x0
        dy = y - y0
        if np.sqrt(dx * dx + dy * dy) > tol:
            has_axis = True
    return 1 if has_axis else 0


@njit(nogil=True, cache=True)
def _polyline_length(polyline_xy):
    total = 0.0
    for i in range(len(polyline_xy) - 1):
        dx = polyline_xy[i + 1, 0] - polyline_xy[i, 0]
        dy = polyline_xy[i + 1, 1] - polyline_xy[i, 1]
        total += np.sqrt(dx * dx + dy * dy)
    return float(total)


@njit(nogil=True, cache=True)
def _project_points_to_polyline(points_xy, polyline_xy):
    """Project [x, y] points to nearest position along a polyline."""
    best_dist2 = np.full(len(points_xy), np.inf, dtype=np.float64)
    best_pos = np.zeros(len(points_xy), dtype=np.float64)
    cumulative = 0.0

    for i in range(len(polyline_xy) - 1):
        start_x = polyline_xy[i, 0]
        start_y = polyline_xy[i, 1]
        vec_x = polyline_xy[i + 1, 0] - start_x
        vec_y = polyline_xy[i + 1, 1] - start_y
        seg_len = np.sqrt(vec_x * vec_x + vec_y * vec_y)
        if seg_len <= 1e-12:
            cumulative += seg_len
            continue
        seg_len2 = seg_len * seg_len
        for point_i in range(len(points_xy)):
            rel_x = points_xy[point_i, 0] - start_x
            rel_y = points_xy[point_i, 1] - start_y
            t = (rel_x * vec_x + rel_y * vec_y) / seg_len2
            t = min(1.0, max(0.0, t))
            projected_x = start_x + t * vec_x
            projected_y = start_y + t * vec_y
            dist_x = points_xy[point_i, 0] - projected_x
            dist_y = points_xy[point_i, 1] - projected_y
            dist2 = dist_x * dist_x + dist_y * dist_y
            if dist2 < best_dist2[point_i]:
                best_dist2[point_i] = dist2
                best_pos[point_i] = cumulative + t * seg_len
        cumulative += seg_len

    for point_i in range(len(points_xy)):
        if not np.isfinite(best_dist2[point_i]):
            raise ValueError('point_projection_failed')
    return best_pos


@njit(nogil=True, cache=True)
def _binned_fluorescence_statistics(values, bin_index, n_bins):
    """Compute the pre-existing per-bin fluorescence summary statistics."""
    median_values = np.full(n_bins, np.nan, dtype=np.float64)
    total_values = np.zeros(n_bins, dtype=np.float64)
    sum_squared_values = np.zeros(n_bins, dtype=np.float64)
    sample_variance_values = np.full(n_bins, np.nan, dtype=np.float64)

    for bin_i in range(n_bins):
        count = 0
        for value_i in range(len(values)):
            if bin_index[value_i] == bin_i:
                count += 1
        if count == 0:
            continue

        bin_values = np.empty(count, dtype=np.float64)
        value_out_i = 0
        for value_i in range(len(values)):
            if bin_index[value_i] == bin_i:
                bin_values[value_out_i] = values[value_i]
                value_out_i += 1

        median_values[bin_i] = np.median(bin_values)
        total = np.sum(bin_values)
        total_values[bin_i] = total
        sum_squared_values[bin_i] = np.sum(bin_values * bin_values)
        if count > 1:
            mean = total / count
            squared_deviation = 0.0
            for value_i in range(count):
                delta = bin_values[value_i] - mean
                squared_deviation += delta * delta
            sample_variance_values[bin_i] = squared_deviation / (count - 1)

    return (median_values, total_values, sum_squared_values,
            sample_variance_values)


# =============================================================================
# Pipeline-level wrapper
# =============================================================================

def process_channel_masks(masks_dict, images_dict, time_points,
                          umppx=0.065, max_cells=50,
                          step_size=None, smooth_tol=0.05,
                          background_fluorescence=None):
    """Process all cells in a channel, matching the Moma_CPM_FOV interface.

    Parameters
    ----------
    masks_dict : dict {time_key: np.ndarray}
        Labeled cell masks per time point.
    images_dict : dict {time_key: {channel: np.ndarray}}
        Channel images per time point.
    time_points : list of str
        Ordered time point keys.
    umppx : float
        Microns per pixel.
    max_cells : int
        Maximum number of cells to process per time point.
    step_size : float or None
    smooth_tol : float
    background_fluorescence : dict or None
        Optional ``{time_key: {channel: background}}`` mapping.

    Returns
    -------
    list of list of CellDT
        Outer list indexed by time point, inner list = cells at that time.
    """
    cells_by_time = []
    for time_key in time_points:
        mask = masks_dict.get(time_key)
        images = images_dict.get(time_key, {})
        time_background = (background_fluorescence or {}).get(time_key, {})
        cells = []
        if mask is not None and mask.max() > 0:
            n_cells = min(int(mask.max()), max_cells)
            for cell_i in range(1, n_cells + 1):
                cell_mask = (mask == cell_i)
                cell = CellDT(cell_index=cell_i, fov=time_key, umppx=umppx)
                cell.compute_from_mask(cell_mask, channel_images=images,
                                       step_size=step_size,
                                       smooth_tol=smooth_tol,
                                       background_fluorescence=time_background)
                cells.append(cell)
        cells_by_time.append(cells)

    # Transpose: [cell_index][time_index] -> [time_index][cell_index]
    # (matches original behavior: cells_list_along_time[time_i] = list of cells)
    return cells_by_time
