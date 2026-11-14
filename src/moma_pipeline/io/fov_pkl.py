"""DT-schema FOV pickle export/load helpers.

Ported from Moma_cellPose/moma_dt_pipeline_utils.py into
moma_pipeline.io.fov_pkl. The morphology imports (describe_dt_morphology,
CellDT) stay function-level so this module imports without pulling numba.
"""

import os
import pickle

from moma_pipeline.io.dt_exports import collect_flu_distribution_dt


def save_flu_distribution_dt(export_dir, fov_name, fov_cells_dict):
    """Save one binary fluorescence-distribution pickle for a FOV."""
    distribution = collect_flu_distribution_dt(fov_cells_dict)
    output_path = os.path.join(export_dir, f'{fov_name}_flu_distribution.pkl')
    with open(output_path, 'wb') as file:
        pickle.dump(distribution, file)
    return output_path, distribution


def export_Moma_fov_data_dt(time_points, opticalChannels, other_channels,
                            cell_detection_opticalChannel_name, channel_keys,
                            fov_cells_mask_dict, fov_channels_image_dict,
                            fov_cells_dict, time_points_seconds,
                            images_background_fluorescence):
    """Export FOV data dict for pickle serialization (DT version)."""
    from moma_pipeline.morphology.worker import describe_dt_morphology

    info_dict = {
        'time_points': time_points,
        'opticalChannels': opticalChannels,
        'other_channels': other_channels,
        'cell_detection_opticalChannel_name': cell_detection_opticalChannel_name,
        'channel_keys': channel_keys,
        'algorithm': 'DT',
        'morphology': describe_dt_morphology(),
    }

    fov_cells_pars_dict = {}
    for sideChannel_key, sideChannel_dict in fov_cells_dict.items():
        fov_cells_pars_dict[sideChannel_key] = {}
        for time_key, cell_list in sideChannel_dict.items():
            fov_cells_pars_dict[sideChannel_key][time_key] = [
                cell.export_cell_parameters() for cell in cell_list
            ]

    cells_sideChannel_image_dict = {
        'fov_cells_mask_dict': fov_cells_mask_dict,
        'fov_channels_image_dict': fov_channels_image_dict,
        'time_points_seconds': time_points_seconds,
        'images_background_fluorescence': images_background_fluorescence,
        'fov_cells_pars_dict': fov_cells_pars_dict,
    }

    return {
        'info_dict': info_dict,
        'cells_sideChannel_image_dict': cells_sideChannel_image_dict,
    }


def load_Moma_fov_data_dt(data_dict):
    """Load FOV data from a DT pickle dict."""
    # Imported lazily so io.fov_pkl imports without numba.
    from moma_pipeline.morphology.celldt import CellDT

    info_dict = data_dict['info_dict']
    cells_sideChannel_image_dict = data_dict['cells_sideChannel_image_dict']

    time_points = info_dict['time_points']
    opticalChannels = info_dict['opticalChannels']
    other_channels = info_dict['other_channels']
    cell_detection_opticalChannel_name = info_dict['cell_detection_opticalChannel_name']
    channel_keys = info_dict['channel_keys']

    fov_cells_mask_dict = cells_sideChannel_image_dict['fov_cells_mask_dict']
    fov_channels_image_dict = cells_sideChannel_image_dict['fov_channels_image_dict']
    time_points_seconds = cells_sideChannel_image_dict['time_points_seconds']
    fov_cells_pars_dict = cells_sideChannel_image_dict['fov_cells_pars_dict']
    images_background_fluorescence = (
        cells_sideChannel_image_dict['images_background_fluorescence']
    )

    fov_cells_dict = {}
    for channel_key in fov_cells_pars_dict:
        fov_cells_dict[channel_key] = {}
        for time_key, cell_pars_list in fov_cells_pars_dict[channel_key].items():
            fov_cells_dict[channel_key][time_key] = []
            for cell_pars in cell_pars_list:
                cell = CellDT()
                cell.load_cell_parameters(cell_pars)
                fov_cells_dict[channel_key][time_key].append(cell)

    return (
        fov_cells_dict, fov_channels_image_dict, fov_cells_mask_dict,
        time_points_seconds, time_points, opticalChannels, other_channels,
        cell_detection_opticalChannel_name, channel_keys,
        images_background_fluorescence,
    )
