import numpy as np
from skimage.draw import polygon, polygon_perimeter
from skimage.morphology import thin

from scr.config.numerics import WP
from scr.utils.types_alias import Contour, Contours, Mask
from scr.geometry.contours.orientation import is_ccw


def surface(
        contours: Contour | Contours,
        shape: tuple[int, int],
        *,
        dtype: type = WP,
) -> Mask:
    """
    Rasterise contours using discrete polygon filling.

    Parameters
    ----------
    contours : sequence of Contour
    shape : tuple of int
    dtype : type, optional

    Returns
    -------
    mask : ndarray
        Float mask in [-len(contours), +len(contours)].
    """
    total = np.zeros(shape, dtype=float)

    for contour in contours:
        if len(contour) < 3:
            continue

        # CCW adds area, CW subtracts it (creates holes)
        sign = 1.0 if is_ccw(contour) else -1.0

        rr, cc = polygon(contour[:, 0], contour[:, 1], shape=shape)
        total[rr, cc] += sign

    return total.astype(dtype)


def border(
        contours: Contours,
        shape: tuple[int, int],
        *,
        dtype: type = WP,
        thin_border: bool = True,
) -> Mask:
    """
    Rasterise contour borders using discrete perimeter.

    Parameters
    ----------
    contours : sequence of Contour
    shape : tuple of int
    dtype : type, optional
    thin_border : bool, optional

    Returns
    -------
    mask : ndarray
        Float mask in [0, 1].
    """
    mask = np.zeros(shape, dtype=bool)

    for contour in contours:
        if len(contour) < 2:
            continue

        rr, cc = polygon_perimeter(contour[:, 0], contour[:, 1], shape=shape, clip=True)
        mask[rr, cc] = True

    if thin_border:
        mask = thin(mask)

    return np.clip(mask, 0.0, 1.0).astype(dtype)
