import numpy as np
import pandas as pd

from scr.config.numerics import WP


def apply_segments_to_combined_df(
        combined_df: pd.DataFrame,
        segments_df: pd.DataFrame,
) -> None:
    """
    Annotate combined_df with segment-wise quantities
    (slope, relative slope, phase duration) in place.
    """

    for col in ["segment_slope", "segment_relative_slope", "phase_duration"]:
        if col not in combined_df:
            combined_df[col] = WP(np.nan)

    # --- group once
    combined_groups = combined_df.groupby(
        ["observation_id", "sunspot_id"],
        observed=True,
        sort=False,
    )

    segment_groups = segments_df.groupby(
        ["observation_id", "sunspot_id"],
        observed=True,
        sort=False,
    )

    # --- iterate only over overlapping keys
    for key, seg_group in segment_groups:
        if key not in combined_groups.groups:
            continue

        idx = combined_groups.groups[key]
        frames = combined_df.loc[idx, "frame"].to_numpy()

        for _, seg in seg_group.iterrows():
            mask = (
                (frames >= np.round(seg["start"])) &
                (frames <= np.round(seg["stop"]))
            )

            if not mask.any():
                continue

            loc = idx[mask]

            combined_df.loc[loc, "segment_slope"] = WP(seg["slope"])
            combined_df.loc[loc, "segment_relative_slope"] = WP(seg["relative_slope"])
            combined_df.loc[loc, "phase_duration"] = WP(seg["duration"])
