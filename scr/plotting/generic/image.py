import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from typing import Literal

from scr.plotting.utils import merge_explicit_kwargs


def plot_image(
        ax: Axes,
        image: np.ndarray,
        *,
        cmap: str = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        origin: Literal["upper", "lower"] = "lower",
        image_kwargs: dict | None = None,
) -> AxesImage:
    """
    Plot a 2D image on given axes.
    """
    image_kwargs = merge_explicit_kwargs(
        image_kwargs,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin=origin,
    )

    im = ax.imshow(
        image,
        **image_kwargs,
    )

    return im
