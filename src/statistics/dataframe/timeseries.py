import pandas as pd
import numpy as np


def add_relative_time(
        df: pd.DataFrame,
        *,
        group_col: str = "spot_global_index",
        frame_col: str = "frame",
        out_col: str = "time_hours",
) -> pd.DataFrame:
    """
    Add a relative time column assuming unit spacing between frames.
    Time is measured from the first appearance within each group.
    """
    min_frames = df.groupby(group_col)[frame_col].transform("min")
    df[out_col] = df[frame_col] - min_frames

    return df


def add_phase_progress(
        df: pd.DataFrame,
        *,
        group_col: str = "spot_global_index",
        frame_col: str = "frame",
        phase_col: str = "phase",
        time_col: str = "time_hours",
) -> pd.DataFrame:
    """
    For each contiguous run of the same phase within an object, add:
      - phase_elapsed_hours: time since that run started
      - phase_duration_hours: total duration of that run
      - phase_progress: phase_elapsed_hours / phase_duration_hours, in [0, 1]

    Handles objects with multiple non-contiguous runs of the same phase
    label (e.g. re-forming after a merge) by resetting elapsed time at
    each new run, rather than using time since the object's first frame.
    """
    df = df.sort_values([group_col, frame_col]).copy()

    same_as_prev = (df[group_col] == df[group_col].shift()) & (df[phase_col] == df[phase_col].shift())
    run_id = (~same_as_prev).cumsum()

    run_start = df.groupby(run_id)[time_col].transform("min")
    run_end = df.groupby(run_id)[time_col].transform("max")

    df["phase_elapsed_hours"] = df[time_col] - run_start
    df["phase_duration_hours"] = run_end - run_start
    with np.errstate(invalid="ignore", divide="ignore"):
        df["phase_progress"] = np.where(
            df["phase_duration_hours"] > 0,
            df["phase_elapsed_hours"] / df["phase_duration_hours"],
            np.nan,   # single-frame runs have no meaningful progress
        )
    return df
