from matplotlib.axes import Axes
from typing import Sequence

from scr.plotting.types import ContourGroup
from scr.plotting.generic.contours import plot_contours


def plot_contour_groups(
        ax: Axes,
        contour_groups: Sequence[ContourGroup],
        *,
        default_contour_kwargs: dict | None = None,
) -> None:
    if default_contour_kwargs is None:
        default_contour_kwargs = {}

    for group in contour_groups:
        style = default_contour_kwargs | group.style

        plot_contours(
            ax,
            group.contours,
            label=group.label,
            contour_kwargs=style,
        )
