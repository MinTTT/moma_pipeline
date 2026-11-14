"""Cell-detection channel auto-detection helpers.

Ported verbatim from Moma_cellPose/moma_dt_pipeline_utils.py into
moma_pipeline.detection.channels.
"""

import os


CELL_DETECTION_NAME_DIR_NAME_LIST = ['BF', 'phase']
# Ordered lowercase folder-name patterns used to auto-detect the
# cell-detection (bright-field / phase-contrast) optical channel when the
# user does not specify one. Earlier patterns win when several folders match;
# exact matches are tried for every pattern before substring matches.
# 'phase' also covers e.g. 'phase3d'; 'brightfield'/'transmitted'/'dic' were
# previously recognized only by the ND2 wrapper (unified 2026-11-01).
CELL_DETECTION_CHANNEL_PATTERNS = ['bf', 'phase', 'brightfield',
                                   'transmitted', 'dic']


def auto_detect_cell_detection_channel(fov_dir, patterns=None):
    """Auto-detect the cell-detection channel subdirectory of a FOV.

    Subdirectory names are matched case-insensitively against 'patterns'
    (default: CELL_DETECTION_CHANNEL_PATTERNS, i.e. 'bf' then 'phase').
    Exact name matches are preferred over substring matches, and earlier
    patterns win over later ones. Ties are broken alphabetically so the
    result is deterministic.

    Returns the matched subdirectory name. Raises ValueError when nothing
    matches, listing the available folders.
    """
    if patterns is None:
        patterns = CELL_DETECTION_CHANNEL_PATTERNS
    subdirs = sorted(
        d for d in os.listdir(fov_dir)
        if os.path.isdir(os.path.join(fov_dir, d))
    )
    if not subdirs:
        raise ValueError(
            f"No optical-channel subdirectories found in FOV directory "
            f"'{fov_dir}'. Specify --cell_detection_channel explicitly."
        )
    lowered = {d: d.lower() for d in subdirs}
    for match_mode in ('exact', 'substring'):
        for pattern in patterns:
            pattern = str(pattern).lower()
            if match_mode == 'exact':
                candidates = [d for d in subdirs if lowered[d] == pattern]
            else:
                candidates = [d for d in subdirs if pattern in lowered[d]]
            if candidates:
                chosen = candidates[0]
                if len(candidates) > 1:
                    print(f"  Multiple channel folders match '{pattern}' "
                          f"{candidates}; using '{chosen}'")
                return chosen
    raise ValueError(
        f"Could not auto-detect a cell-detection channel in '{fov_dir}': "
        f"no folder name matches {patterns}. Available: {subdirs}. "
        f"Specify --cell_detection_channel explicitly."
    )


def resolve_channel_name_in_fov(fov_dir, user_channel):
    """Resolve a channel subdirectory name with case-insensitive matching.

    When 'user_channel' is None or 'auto', the cell-detection channel is
    auto-detected from the FOV's folder names; see
    auto_detect_cell_detection_channel().
    """
    if user_channel is None or str(user_channel).strip().lower() == 'auto':
        matched = auto_detect_cell_detection_channel(fov_dir)
        print(f"  Cell detection channel auto-detected as '{matched}'")
        return matched

    subdirs = [
        d for d in os.listdir(fov_dir)
        if os.path.isdir(os.path.join(fov_dir, d))
    ]
    if user_channel in subdirs:
        return user_channel

    # if user_channel is not found, try case-insensitive match
    lower_to_original = {d.lower(): d for d in subdirs}
    if user_channel.lower() in lower_to_original:
        matched = lower_to_original[user_channel.lower()]
        print(f"  Channel '{user_channel}' matched as '{matched}' (case-insensitive)")
        return matched
    # if user_channel is not found, try to find a channel that in
    # CELL_DETECTION_NAME_DIR_NAME_LIST (case-insensitive)
    for name in CELL_DETECTION_NAME_DIR_NAME_LIST:
        if name.lower() in lower_to_original:
            matched = lower_to_original[name.lower()]
            print(f"  Channel '{user_channel}' not found, using '{matched}' instead")
            return matched

    raise ValueError(
        f"Channel '{user_channel}' not found in FOV directory. "
        f"Available: {subdirs}"
    )
