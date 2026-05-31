import pandas as pd

from scr.utils.types_alias import SunspotsPhasesByObservation, ObjectFilteringMode
from scr.config.filtering import gimme_filtering_kwargs

from scr.io.datasets import load_sunspots_and_df_stat

from scr.statistics.dataframe.filtering import filter_dataset_by_spots
from scr.postanalysis.selection.sunspots import apply_standard_sunspots_phases_filter


def load_filtered_phase_tracks(
        nosuffix_filename: str,
        filtering_options: ObjectFilteringMode | dict,
        filter_phase_tracks: bool = True,
        drop_unknown: bool = True,
) -> tuple[SunspotsPhasesByObservation, pd.DataFrame]:
    """
    Load contour phase tracks and apply standard filtering.

    Returns
    -------
    contours_phases : dict
        Nested contour structure per observation.
    combined_df : pandas.DataFrame
        Filtered metadata table.
    """
    sunspots_phases, combined_df = load_sunspots_and_df_stat(nosuffix_filename)

    if isinstance(filtering_options, str):
        filtering_kwargs = gimme_filtering_kwargs(mode=filtering_options)
    elif isinstance(filtering_options, dict):
        filtering_kwargs = filtering_options
    else:
        raise ValueError(f"Unknown filtering options {filtering_options}.")

    combined_df = filter_dataset_by_spots(
        combined_df,
        filtering_kwargs=filtering_kwargs,
    )

    if drop_unknown and "phase" in combined_df:
        combined_df = combined_df[combined_df["phase"] != "unknown"]

    if filter_phase_tracks:
        sunspots_phases = apply_standard_sunspots_phases_filter(sunspots_phases, combined_df)

    return sunspots_phases, combined_df
