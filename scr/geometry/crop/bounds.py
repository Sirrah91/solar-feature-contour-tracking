import numpy as np

from scr.utils.types_alias import Contour, Contours
from scr.utils.filesystem import is_empty

from scr.geometry.contours.normalization import normalize_contour_input


def compute_crop_bounds(
        contours: Contour | Contours,
        margin: int = 0,
        image_shape: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """
    Compute bounding box covering contours.

    Parameters
    ----------
    contours : list of contours
        Contours defining the main region.
    margin : int
        Extra padding in pixels.
    image_shape : (ny, nx), optional
        If provided, bounds are clipped to image limits.

    Returns
    -------
    (ymin, ymax, xmin, xmax) : tuple of int
        Bounding box coordinates (inclusive-exclusive).
    """
    if is_empty(contours):
        raise ValueError("contours_outer must contain at least one contour")

    contours = normalize_contour_input(contours)

    points = np.vstack(contours)

    y_min, x_min = np.floor(np.min(points, axis=0) - margin)
    y_max, x_max = np.ceil(np.max(points, axis=0) + margin)

    y_min, y_max = int(y_min), int(y_max)
    x_min, x_max = int(x_min), int(x_max)

    if image_shape is not None:
        ny, nx = image_shape
        y_min = max(0, y_min)
        x_min = max(0, x_min)
        y_max = min(ny, y_max)
        x_max = min(nx, x_max)

    return y_min, y_max, x_min, x_max
