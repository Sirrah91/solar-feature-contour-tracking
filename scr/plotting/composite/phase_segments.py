from matplotlib.axes import Axes
import numpy as np

from scr.plotting.generic.lines import plot_line
from scr.plotting.utils import merge_explicit_kwargs


def plot_phase_segments(
        ax: Axes,
        segments: list[tuple[np.ndarray, np.ndarray]],
        x_offset: int | float,
        *,
        alpha: float = 0.07,
        color: str = "gray",
        line_kwargs: dict | None = None,
) -> None:
    """
    Plot phase-evolution segments with a fixed x-offset.
    """
    line_kwargs = merge_explicit_kwargs(
        line_kwargs,
        alpha=alpha,
        color=color,
    )

    for t, y in segments:
        plot_line(
            ax,
            t + x_offset,
            y,
            line_kwargs=line_kwargs,
        )
