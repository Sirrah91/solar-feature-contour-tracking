import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

from scr.plotting.utils import merge_explicit_kwargs


def plot_scatter(
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        *,
        scatter_kwargs: dict | None = None,
) -> PathCollection:
    """
    Plot a scatter plot with finite-value filtering.
    """
    scatter_kwargs = merge_explicit_kwargs(
        scatter_kwargs
    )

    mask = np.isfinite(x) & np.isfinite(y)

    sc = ax.scatter(
        x[mask],
        y[mask],
        **scatter_kwargs,
    )
    return sc
