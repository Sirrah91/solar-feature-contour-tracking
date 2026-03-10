import numpy as np
import pandas as pd
from os import path

from scr.config.paths import SLOPES_BASENAME
from scr.config.numerics import WP

from scr.utils.types_alias import SunspotsPhasesByObservation
from scr.utils.nested import nested_cast_arrays_dtype

from scr.io.parquet import load_parquet
from scr.io.sunspots import load_sunspot_file

from scr.statistics.dataframe.flatten import flatten_spot_features_with_frame
from scr.statistics.dataframe.filtering import filter_combined_df
from scr.statistics.segments.annotation import apply_segments_to_combined_df
from scr.statistics.segments.collection import collect_slopes

from scr.postanalysis.phases import split_by_phase


def compute_phase_split(
        stats_paths: list[str],
        slope_path: str = SLOPES_BASENAME,
        collect_new_slopes: bool = False,
) -> tuple[SunspotsPhasesByObservation, pd.DataFrame]:
    # --------------------------------------------------------------
    # 1) Load all tracks & statistics
    # --------------------------------------------------------------

    print("Collecting statistics and contours...")

    all_stats: dict = {}
    all_sunspots: dict = {}
    all_filenames: dict = {}

    for stats_path in stats_paths:
        sunspots, stats, metadata, events = load_sunspot_file(stats_path)
        all_stats[stats_path] = stats
        all_sunspots[stats_path] = sunspots
        all_filenames[stats_path] = metadata["image_paths"]

    # --------------------------------------------------------------
    # 2) Flatten statistics
    # --------------------------------------------------------------

    print("Flatten statistics...")

    combined_df = flatten_spot_features_with_frame(all_stats=all_stats)

    combined_df["image_path"] = [
        all_filenames[id_][frame]
        for id_, frame in zip(combined_df["observation_id"], combined_df["frame"])
    ]
    combined_df["image_path"] = combined_df["image_path"].astype("category")

    if collect_new_slopes or not path.isfile(slope_path):
        # --------------------------------------------------------------
        # 3) Prefilter ONLY for slope fitting
        # --------------------------------------------------------------

        print("Slope fitting...")

        df_fit = filter_combined_df(
            df=combined_df,
            filtering_kwargs={
                "sunspot_mu_min": {"min_value": 0.15, "mode": "frame-wise"}
            }
        )

        df_fit = df_fit[
            [
                "observation_id",
                "sunspot_id",
                "spot_global_index",
                "frame",
                "Br_sunspot_flux_corr_total",
            ]
        ].copy()

        # --------------------------------------------------------------
        # 4) Fit slopes
        # --------------------------------------------------------------

        segments_df = collect_slopes(
            df=df_fit,
            slope_path=slope_path,
            control_plots=True,
        )
    else:
        print("Using precomputed slopes...")

        segments_df = load_parquet(slope_path)

    # --------------------------------------------------------------
    # 5) Apply segments to combined_df
    # --------------------------------------------------------------

    print("Merging statistics and segments...")

    apply_segments_to_combined_df(
        combined_df=combined_df,
        segments_df=segments_df,
    )

    # --------------------------------------------------------------
    # 6) Phase assignment (forming / stable / decaying)
    # --------------------------------------------------------------

    print("Phase assignment...")

    master_column = "segment_slope"
    slope_threshold = 0.00225

    conditions = [
        np.isfinite(combined_df[master_column]) &
        (combined_df[master_column] > slope_threshold),

        np.isfinite(combined_df[master_column]) &
        (combined_df[master_column] <= slope_threshold) &
        (combined_df[master_column] >= -slope_threshold),

        np.isfinite(combined_df[master_column]) &
        (combined_df[master_column] < -slope_threshold),
    ]

    choices = ["forming", "stable", "decaying"]

    combined_df["phase"] = np.select(
        conditions,
        choices,
        default="unknown",
    )

    combined_df["phase"] = combined_df["phase"].astype("category")

    # --------------------------------------------------------------
    # 7) Split back by phase
    # --------------------------------------------------------------

    print("Splitting by phases...")

    sunspots_phases = split_by_phase(
        combined_df,
        all_sunspots,
    )

    sunspots_phases = nested_cast_arrays_dtype(sunspots_phases, dtype=WP)

    return sunspots_phases, combined_df
