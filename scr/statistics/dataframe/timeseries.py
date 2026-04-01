import pandas as pd


def add_relative_time(
        df: pd.DataFrame,
        *,
        frame_col: str = "frame",
        out_col: str = "time_hours",
) -> pd.DataFrame:
    """
    Add a relative time column assuming unit spacing between frames.
    Time is measured from first appearance.
    """
    df = df.copy()
    df[out_col] = df[frame_col] - df[frame_col].min()
    return df
