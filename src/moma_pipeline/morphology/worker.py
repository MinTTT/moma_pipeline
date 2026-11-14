"""Loky worker entry point for DT morphology.

Ported from Moma_cellPose/moma_dt_worker.py into moma_pipeline.morphology.

This module must stay LIGHTWEIGHT: loky/cloudpickle serializes functions in
importable modules by reference, so every worker process imports the module
that defines the task function. Keeping the worker here (numpy/numba/scipy
only, via moma_pipeline.morphology.celldt) avoids pulling torch/cellpose/cv2 into
dozens of worker processes, which otherwise costs ~1-3 GB RSS per worker and
long import storms on many-core servers.
"""

import numpy as np

PRODUCTION_SPINE_MODE = "channel_axis"
PRODUCTION_CHANNEL_AXIS = (0.0, 1.0)
PRODUCTION_MORPHOLOGY_VERSION = "dt_channel_reference_bounded_poles_2026_11_06"
# Retained import alias for callers of the former opt-in candidate.
CONTOUR_POLE_MORPHOLOGY_VERSION = PRODUCTION_MORPHOLOGY_VERSION


def describe_dt_morphology(pole_refinement="contour"):
    """Describe frozen initial references and bounded pole fitting for all cells.

    "none" is a deprecated compatibility choice and describes the same current
    behavior as "contour"; it cannot disable channel refinement.
    """
    if pole_refinement not in {"none", "contour"}:
        raise ValueError("pole_refinement must be none or contour")
    from .celldt import DEFAULT_UM_PER_PX
    from .voronoi_dt import (
        DEFAULT_STEP_SIZE, DEFAULT_SMOOTH_TOL, CHANNEL_AXIS_SMOOTH_WINDOW,
        MAX_BODY_ITERATIONS, BODY_CONVERGENCE_TOL_PX, BODY_UPDATE_FRACTION,
    )
    return {
        "version": PRODUCTION_MORPHOLOGY_VERSION,
        "method_version": "7.0.0",
        "method": "contour_pipeline",
        "spine_mode": PRODUCTION_SPINE_MODE,
        "spine_method": "channel_axis_path_only",
        "channel_axis": list(PRODUCTION_CHANNEL_AXIS),
        "automatic_step_px": 1.0,
        "smooth_tol_applied": True,
        "centerline_smoothing_vertices": CHANNEL_AXIS_SMOOTH_WINDOW,
        "initial_reference": {
            "stage": "first_valid_contour_mesh_before_midpoint_updates",
            "saved_geometry": "mesh_poles_radii_length_bounds_and_local_contour",
            "pole_tips": "local_first_inside_to_outside_raw_contour_exits",
            "cap_center": "tip_minus_frozen_radius_times_outward",
            "terminal_direction": "chord_across_nearest_up_to_5_body_centers",
            "frozen_during_refinement": True,
            "diagnostic_field": "reference_geometry",
            "diagnostic_fields": [
                "length_px", "endpoint_radii", "poles", "length_bounds_px",
                "center_search_bounds_px", "terminal_contour_segments", "body_rib_count",
            ],
            "diagnostic_units": "pixels",
            "terminal_contour_segment_shape": "two_lists_of_Nx4_xy_endpoints",
        },
        "body_refinement": "contour_rib_midpoint_iterations",
        "body_refinement_details": {
            "max_iterations": MAX_BODY_ITERATIONS,
            "max_attempts": 1,
            "convergence_statistic": "mean_matched_spine_shift_px",
            "convergence_threshold_px": BODY_CONVERGENCE_TOL_PX,
            "beta": BODY_UPDATE_FRACTION,
            "iteration_limit": "best_valid_coverage_preserving_body",
            "endpoint_neighborhoods": "fixed_seed_terminals_all_cells",
            "branch_endpoint_neighborhoods": "fixed_seed_terminals",
            "coverage_required": True,
            "endpoint_stability_required": True,
        },
        "pole_baseline": "initial_contour_body_local_first_exit_caps",
        "rib_halfwidth": "contour_paired_exits_body_conditional_pole_radius",
        "pole_refinement": "contour",
        "pole_refinement_mandatory": "all_except_dedicated_round_short_routes",
        "branch_policy": "independent_local_radius_bounded_fit",
        "branch_measurement_scope": "selected_primary_body_mesh;whole_mask_raw_area_and_fluorescence",
        "branch_detection": {
            "method": "substantial_unused_skeleton_arm",
            "minimum_skeleton_endpoints": 3,
            "off_route_arm_threshold_median_raw_dt": 2.0,
            "median_raw_dt_population": "selected_skeleton_route",
        },
        "pole_failure": "exact_initial_reference_fallback",
        "terminal_rib_removal": "mandatory",
        "step_size": DEFAULT_STEP_SIZE,
        "smooth_tol": DEFAULT_SMOOTH_TOL,
        "umppx": DEFAULT_UM_PER_PX,
        "pole_refinement_details": {
            "status": "production",
            "scope": "channel_contour_body_and_terminal_poles_all_cells",
            "contour": "raw_mask_marching_squares_0.5",
            "initial_path": "channel_axis_selected_route_only",
            "body_refinement": "contour_rib_midpoint_spine_iterations_fixed_seed_endpoints",
            "cap_radius": "shared_initial_radius_if_unbranched_independent_initial_local_radii_if_branched",
            "cap_radius_shared": "unbranched_only",
            "cap_radius_frozen_during_fitting": True,
            "cap_radius_estimator": "median_valid_contour_body_rib_halfwidth",
            "cap_radius_population": "valid_initial_contour_body_ribs_before_midpoint_updates_and_cap_trimming",
            "cap_radius_frozen_during_fitting_applies_to": "all_cells",
            "cap_radius_excludes_cap_rows": True,
            "cap_radius_central_trim": False,
            "cap_radius_estimator_applies_to": "unbranched_only",
            "branch_skip_pole_refinement": False,
            "branch_radius_estimator": "median_terminal_valid_initial_body_rib_halfwidth",
            "branch_radius_population": "valid_terminal_initial_contour_body_ribs_before_midpoint_updates_and_cap_trimming",
            "branch_radius_terminal_fraction": 0.20,
            "branch_radius_min_ribs": 3,
            "branch_radius_max_ribs": 21,
            "branch_cap_center": "local_first_contour_exit_minus_frozen_local_radius",
            "cap_center": "independently_optimized_scalar_spine_position_in_reference_bounds",
            "terminal_direction": "final_spine_tangent_consistent_with_profile_tip",
            "cap_fit": "fixed_radius_circular_caps_against_frozen_local_raw_contour_all_cells",
            "local_contour": "contiguous_ordered_arc_within_2R_of_initial_tip_and_forward_of_reference_cap_center",
            "local_contour_frozen": True,
            "center_search": "inward_from_each_saved_reference_cap_center",
            "center_search_limit_px": "max(1,min(0.5R,0.1_reference_center_arc))",
            "tip_displacement_limit_px": "max(1,0.5R)",
            "length_bounds_px": "reference_length_plus_or_minus_sum_max(1,0.5R)_per_end",
            "cap_residual_qc": "soft_warning_fit_low_confidence",
            "pole_failure": "exact_initial_reference_fallback",
            "terminal_rib_removal": "mandatory_before_morphology",
            "validation": "strict_topology_noncrossing_finite_order_centers_inside_final_tangent_profile_consistency_reference_length_and_tip_coverage",
            "round_short": "existing_geometry_not_applicable",
        },
        "acceptance_constraints": {
            "body": "ordered_selected_route_coverage_and_endpoint_stability",
            "final": "finite_strict_noncrossing_ordered_centers_inside_reference_length_bounds_and_tip_neighborhoods",
            "tip_identity_tolerance_px": 1e-6,
            "fallback": "exact_initial_saved_reference_on_any_final_failure",
        },
    }


def dt_timepoint_compute(cell_masks, c_images, background_fluorescence,
                         max_cell_number, distribution_channels,
                         flu_distribution_bins, time_key, CellDT_class=None, *,
                         umppx=None, cell_indices=None, spine_mode=None,
                         channel_axis=None, pole_refinement=None):
    """Compute CellDT morphology for all cells of one channel/timepoint.

    Module-level and free of instance state so calls carrying only plain
    picklable data can be sent to process-based (loky) workers. Returns the
    list of CellDT objects. ``umppx=None`` uses the production scale. An
    explicit ordered iterable of positive integer ``cell_indices`` preserves
    its IDs and order, including failed cells for labels absent from the mask,
    and bypasses ``max_cell_number``. With no explicit IDs, the production
    numeric-label traversal and cap are unchanged. Moma production uses
    channel_axis along image +y; explicit spine_mode="default" reproduces the
    previous contour-refined algorithm. Channel-axis calls always run contour
    body refinement in one attempt of at most 15 iterations (beta=0.45; mean
    matched spine shift <0.25 px), fixing both seed terminals for every cell.
    Before updates, save the first valid contour mesh and first-exit poles.
    Freeze its median body-rib halfwidth for rods or terminal 20% local medians
    for branches (3 to 21 ribs per end). Every ordinary cell then fits its pole
    centers within saved reference bounds against frozen local contour arcs.
    Strict validation, endpoint coverage and reference length bounds must pass;
    any final failure restores the exact initial saved reference. Dedicated
    round/short routes remain not_applicable. "none" is a deprecated equivalent.
    """
    if CellDT_class is None:
        from .celldt import CellDT as CellDT_class

    requested_refinement = "none" if pole_refinement is None else pole_refinement
    morphology = describe_dt_morphology(requested_refinement)
    selected_mode = morphology["spine_mode"] if spine_mode is None else spine_mode
    selected_axis = morphology["channel_axis"] if channel_axis is None else channel_axis
    if requested_refinement == "contour" and selected_mode != "channel_axis":
        raise ValueError("contour pole_refinement requires spine_mode=channel_axis")
    refinement_kwargs = ({"pole_refinement": "contour"}
                         if selected_mode == "channel_axis" else {})
    scale = morphology["umppx"]
    if umppx is not None:
        try:
            scale = float(umppx)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("umppx must be finite and positive") from exc
        if isinstance(umppx, (bool, np.bool_)) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("umppx must be finite and positive")

    if cell_indices is not None:
        if isinstance(cell_indices, (str, bytes, set, frozenset, dict)):
            raise ValueError("cell_indices must be an ordered iterable of positive unique integers")
        try:
            indices = list(cell_indices)
        except TypeError as exc:
            raise ValueError("cell_indices must be an ordered iterable of positive unique integers") from exc
        if any(isinstance(index, (bool, np.bool_)) or
               not isinstance(index, (int, np.integer)) or index <= 0
               for index in indices):
            raise ValueError("cell_indices must contain positive unique integers")
        indices = [int(index) for index in indices]
        if len(set(indices)) != len(indices):
            raise ValueError("cell_indices must contain positive unique integers")
        if indices and cell_masks is None:
            raise ValueError("cell_masks is required when cell_indices are requested")
    elif cell_masks is not None and int(np.max(cell_masks)) > 0:
        n_cells = min(int(np.max(cell_masks)), max_cell_number)
        indices = range(1, n_cells + 1)
    else:
        indices = []

    cells = []
    for cell_i in indices:
        cell_mask = cell_masks == cell_i
        cell = CellDT_class(cell_index=cell_i, fov=time_key, umppx=scale)
        cell.compute_from_mask(
            cell_mask,
            channel_images=c_images,
            step_size=morphology["step_size"],
            smooth_tol=morphology["smooth_tol"],
            spine_mode=selected_mode,
            channel_axis=selected_axis,
            flu_distribution_channels=distribution_channels,
            flu_distribution_bins=flu_distribution_bins,
            background_fluorescence=background_fluorescence,
            **refinement_kwargs,
        )
        cells.append(cell)
    return cells
