import pandas as pd


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
