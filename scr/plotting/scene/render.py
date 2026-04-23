import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from typing import Sequence, Callable, Literal

from scr.plotting.types import ContourGroup
from scr.plotting.generic.image import plot_image
from scr.plotting.composite.contours import plot_contour_groups


def render_scene(
        ax: Axes,
        *,
        image: np.ndarray | None = None,
        contour_groups: Sequence[ContourGroup] = (),
        cmap: str = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        origin: Literal["upper", "lower"] = "lower",
        image_kwargs: dict | None = None,
        contour_kwargs: dict | None = None,
        annotations: Callable[[Axes, Sequence[ContourGroup]], None] | None = None,
        return_image: bool = False,
) -> AxesImage | None:
    """
    Render a single plotting scene on one Axes.

    This is a pure composition helper.
    """
    im = None

    if image is not None:
        im = plot_image(
            ax,
            image,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin=origin,
            image_kwargs=image_kwargs,
        )

    if contour_groups:
        plot_contour_groups(ax, contour_groups, default_contour_kwargs=contour_kwargs)

    if annotations is not None:
        annotations(ax, contour_groups)

    if im is not None:
        ax.set_xlim(im.get_extent()[:2])
        ax.set_ylim(im.get_extent()[2:])

    if return_image:
        return im
