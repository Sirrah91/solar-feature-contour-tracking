import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def extract_phase_segments(
        df: pd.DataFrame,
        phase_name: str,
        quantity_col: str,
        min_length: int = 5,
        n_interp: int = 30,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Extract contiguous segments of a given phase and interpolate them
    to a common normalised time axis [0, 1].
    """
    segments = []
    t_common = np.linspace(0, 1, n_interp)

    for _, g in df.groupby("spot_global_index", observed=True):
        g = g.sort_values("frame")

        is_phase = g["phase"] == phase_name
        breaks = is_phase.ne(is_phase.shift()).cumsum()

        for _, seg in g[is_phase].groupby(breaks, observed=True):
            if len(seg) < min_length:
                continue

            t = seg["frame"].to_numpy()
            y = seg[quantity_col].to_numpy()

            t_norm = (t - t[0]) / (t[-1] - t[0])

            f = interp1d(
                t_norm,
                y,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            segments.append((t_common, f(t_common)))

    return segments


def median_curve(
        segments: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    return np.nanmedian([y for _, y in segments], axis=0)
