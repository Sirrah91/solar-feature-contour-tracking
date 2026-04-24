import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from typing import Sequence, Literal

from scr.plotting.types import ContourGroup
from scr.plotting.generic.image import plot_image
from scr.plotting.composite.contours import plot_contour_groups


def plot_image_with_contours(
        ax: Axes,
        image: np.ndarray,
        *,
        contour_groups: Sequence[ContourGroup] = (),
        cmap: str = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        origin: Literal["upper", "lower"] = "lower",
        image_kwargs: dict | None = None,
) -> AxesImage:
    """
    Plot an image with one or more contour groups overlaid.
    """
    im = plot_image(
        ax,
        image,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        image_kwargs=image_kwargs,
    )

    plot_contour_groups(ax, contour_groups=contour_groups)

    ax.set_xlim(im.get_extent()[:2])
    ax.set_ylim(im.get_extent()[2:])

    return im
