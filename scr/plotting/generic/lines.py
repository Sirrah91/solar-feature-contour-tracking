import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from scr.plotting.utils import merge_explicit_kwargs


def plot_line(
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        *,
        line_kwargs: dict | None = None,
) -> Line2D:
    """
    Plot a line with finite-value filtering.
    """
    line_kwargs = merge_explicit_kwargs(
        line_kwargs
    )

    mask = np.isfinite(x) & np.isfinite(y)

    line, = ax.plot(
        x[mask],
        y[mask],
        **line_kwargs,
    )
    return line
