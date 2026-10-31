import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from src.plotting.utils import merge_explicit_kwargs


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


def plot_line_with_dashed_gaps(
        ax,
        x,
        y,
        *,
        color: str = "black",
        linewidth: float = 2.0,
        line_kwargs: dict | None = None,
) -> None:
    """Plot a curve with solid valid segments and dashed gaps."""
    line_kwargs = merge_explicit_kwargs(
        line_kwargs,
        color=color,
        linewidth=linewidth,
    )

    valid = np.isfinite(x) & np.isfinite(y)

    # Find contiguous runs of equal validity.
    changes = np.flatnonzero(np.diff(valid.astype(int))) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(valid)]

    for start, end in zip(starts, ends):
        if valid[start]:
            ax.plot(
                x[start:end],
                y[start:end],
                linestyle="-",
                **line_kwargs,
            )
        else:
            # Need valid points immediately before/after the gap
            if start > 0 and end < len(valid):
                ax.plot(
                    x[[start - 1, end]],
                    y[[start - 1, end]],
                    linestyle="--",
                    **line_kwargs
                )
