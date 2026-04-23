import numpy as np
from skimage.draw import polygon_perimeter
from skimage.morphology import thin

from scr.utils.types_alias import Contour, Contours, Mask
from scr.geometry.contours.normalization import normalize_contour_input


def contours_to_border_mask(
    contours: Contour | Contours,
    shape: tuple[int, int],
    thin_border: bool = True,
) -> Mask:
    """
    Rasterise contour edges into a 1-pixel border mask.
    """
    contours = normalize_contour_input(contours)
    mask = np.zeros(shape, dtype=bool)

    for contour in contours:
        r, c = contour[:, 0], contour[:, 1]
        rr, cc = polygon_perimeter(r, c, shape=shape, clip=True)
        mask[rr, cc] = True

    if thin_border:
        mask = thin(mask)

    return mask
