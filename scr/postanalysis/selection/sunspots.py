import pandas as pd

from scr.utils.collections import nested_defaultdict
from scr.utils.types_alias import Sunspots, SunspotsPhases, ObservationID


def apply_standard_sunspot_filter(
        sunspots: dict[ObservationID, Sunspots],
        filtered_df: pd.DataFrame,
) -> dict[ObservationID, Sunspots]:
    """
    Select Sunspots entries consistent with filtered_df rows.

    Parameters
    ----------
    sunspots : dict
        Nested contour dictionary:
        contours[obs][spot][frame] -> data
    filtered_df : pandas.DataFrame
        Filtered metadata table containing
        observation_id, sunspot_id, frame

    Returns
    -------
    dict
        Sunspots pruned to exactly those appearing in filtered_df
    """

    required = {"observation_id", "sunspot_id", "frame"}
    missing = required - set(filtered_df.columns)
    if missing:
        raise KeyError(f"filtered_df missing columns: {missing}")

    # Build fast lookup: (obs, spot, phase) -> set(frames)
    frame_index = (
        filtered_df
        .groupby(["observation_id", "sunspot_id"], observed=True)["frame"]
        .apply(set)
    )

    selected = nested_defaultdict(factory=list, depth=4)

    for obs_id, spots in sunspots.items():
        for spot_id, spot in spots.items():

            key = (obs_id, spot_id)
            if key not in frame_index:
                continue

            allowed_frames = frame_index[key]

            for region, frames in spot.items():
                for frame, data in frames.items():
                    if frame in allowed_frames:
                        selected[obs_id][spot_id][region][frame].extend(data)

    return selected


def apply_standard_sunspots_phases_filter(
        sunspots_phases: dict[ObservationID, SunspotsPhases],
        filtered_df: pd.DataFrame,
) -> dict[ObservationID, SunspotsPhases]:
    """
    Select SunspotsPhases entries consistent with filtered_df rows.

    Parameters
    ----------
    sunspots_phases : dict
        Nested contour dictionary:
        contours[obs][spot][phase][frame] -> data
    filtered_df : pandas.DataFrame
        Filtered metadata table containing
        observation_id, sunspot_id, phase, frame

    Returns
    -------
    dict
        SunspotsPhases pruned to exactly those appearing in filtered_df
    """

    required_cols = {"observation_id", "sunspot_id", "phase", "frame"}
    missing = required_cols - set(filtered_df.columns)
    if missing:
        raise KeyError(f"filtered_df missing columns: {missing}")

    # Build fast lookup: (obs, spot, phase) -> set(frames)
    frame_index = (
        filtered_df
        .groupby(["observation_id", "sunspot_id", "phase"], observed=True)["frame"]
        .apply(set)
    )

    selected = nested_defaultdict(factory=list, depth=5)

    for obs_id, spots in sunspots_phases.items():
        for spot_id, phases in spots.items():
            for phase, spot in phases.items():

                key = (obs_id, spot_id, phase)
                if key not in frame_index:
                    continue

                allowed_frames = frame_index[key]

                for region, frames in spot.items():
                    for frame, data in frames.items():
                        if frame in allowed_frames:
                            selected[obs_id][spot_id][phase][region][frame].extend(data)

    return selected
