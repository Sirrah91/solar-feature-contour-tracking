import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm

from src.config.paths import SLOPES_BASENAME
from src.config.numerics import WP

from src.utils.types_alias import SunspotsPhasesByObservation
from src.utils.filesystem import mkdir, is_empty
from src.utils.nested import nested_cast_arrays_dtype

from src.io.parquet import load_parquet, save_parquet
from src.io.sunspots import load_sunspot_file

from src.statistics.dataframe.flatten import flatten_spot_features_with_frame
from src.statistics.dataframe.filtering import filter_combined_df
from src.statistics.segments.annotation import apply_segments_to_combined_df
from src.statistics.segments.collection import collect_slopes

from src.postanalysis.phases import split_by_phase

from src.utils.logging import get_logger

logger = get_logger("pipelines.processing.split_to_phases")


def compute_phase_split(
        stats_paths: list[str],
        slope_path: str = SLOPES_BASENAME,
        collect_new_slopes: bool = False,
        collect_control_plots: bool = False,
) -> tuple[SunspotsPhasesByObservation, pd.DataFrame]:
    # --------------------------------------------------
    # PASS 1: slopes (streaming)
    # --------------------------------------------------
    if collect_new_slopes or not path.isfile(slope_path):

        logger.info("Slope fitting (streaming)...")

        segments_list = []

        for stats_path in tqdm(stats_paths, desc="Fitting slopes", unit="file"):
            sunspots, stats, metadata, events = load_sunspot_file(stats_path)

            if is_empty(stats):
                continue

            df = flatten_spot_features_with_frame({stats_path: stats})

            df = df[[
                "observation_id",
                "sunspot_id",
                "spot_global_index",
                "frame",
                "Br_sunspot_flux_corr_total",
                "sunspot_mu_min",
            ]]

            df = filter_combined_df(
                df=df,
                filtering_kwargs={
                    "sunspot_mu_min": {"min_value": 0.15, "mode": "frame-wise"}
                }
            )

            segments = collect_slopes(
                df=df,
                slope_path=None,  # do not save the individual slope files
                control_plots=collect_control_plots,
            )
            segments_list.append(segments)

            del df, stats

        segments_df = pd.concat(segments_list, ignore_index=True)

        mkdir(slope_path, is_file_path=True)
        save_parquet(filename=slope_path, df=segments_df)

    else:
        segments_df = load_parquet(slope_path)

    # --------------------------------------------------
    # PASS 2: full dataset (once)
    # --------------------------------------------------
    logger.info("Building combined dataframe...")

    all_stats = {}
    all_sunspots = {}
    all_filenames = {}

    for stats_path in stats_paths:
        sunspots, stats, metadata, events = load_sunspot_file(stats_path)
        all_stats[stats_path] = stats
        all_sunspots[stats_path] = sunspots
        all_filenames[stats_path] = metadata["image_paths"]

    combined_df = flatten_spot_features_with_frame(all_stats)
    del all_stats

    # lighter than strings
    combined_df["image_path"] = [
        all_filenames[id_][frame]
        for id_, frame in zip(combined_df["observation_id"], combined_df["frame"])
    ]
    combined_df["image_path"] = combined_df["image_path"].astype("category")
    del all_filenames

    # --------------------------------------------------------------
    # 3) Apply segments to combined_df
    # --------------------------------------------------------------

    logger.info("Merging statistics and segments...")

    apply_segments_to_combined_df(
        combined_df=combined_df,
        segments_df=segments_df,
    )

    # --------------------------------------------------------------
    # 4) Phase assignment (forming / stable / decaying)
    # --------------------------------------------------------------

    logger.info("Phase assignment...")
    master_column = "segment_slope"
    slope_threshold = 0.00225

    slope = combined_df[master_column]

    combined_df["phase"] = np.select(
        [
            np.isfinite(slope) & (slope > slope_threshold),
            np.isfinite(slope) & (np.abs(slope) <= slope_threshold),
            np.isfinite(slope) & (slope < -slope_threshold),
        ],
        ["forming", "stable", "decaying"],
        default="unknown"
    )
    combined_df["phase"] = combined_df["phase"].astype("category")

    # --------------------------------------------------------------
    # 5) Split back by phase
    # --------------------------------------------------------------

    logger.info("Splitting by phases...")

    sunspots_phases = split_by_phase(
        combined_df,
        all_sunspots,
    )

    sunspots_phases = nested_cast_arrays_dtype(sunspots_phases, dtype=WP)

    return sunspots_phases, combined_df
