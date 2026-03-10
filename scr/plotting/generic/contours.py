import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scr.plotting.utils import merge_explicit_kwargs

from scr.utils.types_alias import Contours


def plot_contours(
        ax: Axes,
        contours: Contours,
        *,
        label: str | None = None,
        contour_kwargs: dict | None = None,
) -> list[Line2D]:
    """
    Plot precomputed contours.

    Each contour must have shape (N, 2) with columns (row, col).
    """
    contour_kwargs = merge_explicit_kwargs(
        contour_kwargs,
    )

    handles: list[Line2D] = []

    for i, contour in enumerate(contours):
        if np.ndim(contour) != 2 or np.shape(contour)[1] != 2:
            raise ValueError("Each contour must have shape (N, 2)")

        h, = ax.plot(
            contour[:, 1],  # x
            contour[:, 0],  # y
            label=label if i == 0 else None,
            **contour_kwargs,
        )
        handles.append(h)

    return handles
