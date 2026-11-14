"""Lightweight DT export helpers shared by pipelines and offline tools.

Only pandas is imported here; CellDT-like objects are accepted without loading
Cellpose, Torch, or the image-processing pipeline. Traversal and column order
match the production export contract.

Ported verbatim from Moma_cellPose.moma_dt_exports into
moma_pipeline.io.dt_exports.
"""

import pandas as pd


def collect_flu_distribution_dt(fov_cells_dict):
    """Collect CellDT fluorescence distributions into a nested FOV dict."""
    fov_distribution = {}
    for channel_key, time_dict in fov_cells_dict.items():
        channel_distribution = {}
        for time_key, cells in time_dict.items():
            time_distribution = {}
            for cell in cells:
                cell_distribution = getattr(cell, 'flu_distribution', {})
                if cell_distribution:
                    time_distribution[int(cell.cell_index)] = cell_distribution
            if time_distribution:
                channel_distribution[time_key] = time_distribution
        if channel_distribution:
            fov_distribution[channel_key] = channel_distribution
    return fov_distribution


def export_cells_to_dataframe_dt(fov_cells_dict, time_points_seconds,
                                 opticalChannels, fluorescence_bac):
    """Export all CellDT parameters to a DataFrame with DT-specific columns."""
    cells_df_columns = [
        'time_s', 'time_point', 'channels', 'cell_index',
        'geometry_mode', 'spine_length', 'area', 'mask_area', 'umppx',
        'width', 'width_area', 'x_min', 'x_max', 'y_min', 'y_max'
    ]
    dt_columns = [
        'curvature', 'volume_um3', 'max_width_um', 'n_ribs', 'pole_radius_um'
    ]
    cells_df_columns += dt_columns

    fluorescent_levels_columns = []
    for channel in opticalChannels:
        fluorescent_levels_columns.append(f'{channel}_background_mean')
        fluorescent_levels_columns.append(f'{channel}_background_median')
        fluorescent_levels_columns.append(f'{channel}_mean')
        fluorescent_levels_columns.append(f'{channel}_median')
        fluorescent_levels_columns.append(f'{channel}_total')
        fluorescent_levels_columns.append(f'{channel}_raw_total')
    cells_df_columns += fluorescent_levels_columns

    cells_data = []
    cell_keys = [
        'cell_index', 'geometry_mode', 'spine_length', 'area', 'mask_area',
        'umppx', 'width', 'width_area', 'x_min', 'x_max', 'y_min', 'y_max'
    ]

    for channel_key in fov_cells_dict:
        for time_key in fov_cells_dict[channel_key]:
            for cell in fov_cells_dict[channel_key][time_key]:
                try:
                    time_in_second = time_points_seconds[time_key]
                except KeyError:
                    time_in_second = None
                cell_data = [time_in_second, time_key, channel_key]

                for key in cell_keys:
                    cell_data.append(getattr(cell, key, None))

                umppx = getattr(cell, 'umppx', 0.065)
                cell_data.append(getattr(cell, '_curvature', None))
                cell_data.append(getattr(cell, '_volume_px', 0.0) * umppx ** 3)
                cell_data.append(getattr(cell, '_max_width_px', 0.0) * umppx)
                cell_data.append(getattr(cell, '_n_ribs', None))
                cell_data.append(getattr(cell, '_pole_radius', 0.0) * umppx)

                for channel in opticalChannels:
                    try:
                        cell_data.append(fluorescence_bac[time_key][channel]['mean'])
                    except KeyError:
                        cell_data.append(None)
                    try:
                        cell_data.append(fluorescence_bac[time_key][channel]['median'])
                    except KeyError:
                        cell_data.append(None)
                    try:
                        cell_data.append(cell.flu_level[channel]['mean'])
                    except KeyError:
                        cell_data.append(None)
                    try:
                        cell_data.append(cell.flu_level[channel]['median'])
                    except KeyError:
                        cell_data.append(None)
                    try:
                        cell_data.append(cell.flu_level[channel]['total'])
                    except KeyError:
                        cell_data.append(None)
                    try:
                        # .get() keeps legacy pickles whose flu_level predates
                        # raw_total exportable without erroring.
                        cell_data.append(cell.flu_level[channel].get('raw_total'))
                    except KeyError:
                        cell_data.append(None)

                cells_data.append(cell_data)

    return pd.DataFrame(cells_data, columns=cells_df_columns)


