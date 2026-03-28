import numpy as np
from skimage.measure import find_contours as _find_contours

from scr.utils.types_alias import Contours

from scr.geometry.contours.filtering import filter_contours_by_area
from scr.geometry.contours.area import contour_area


def find_contours(
        image: np.ndarray,
        level: float
) -> Contours:
    """
    Extract contours from an image at the given threshold level.

    Parameters:
        image: 2D array from which contours are extracted.
        level: Threshold level to extract contours.

    Returns:
        List of contour arrays, each of shape (N, 2) in (y, x) format.
    """
    return _find_contours(np.abs(image.astype(float)), level=level)


def extract_frame_contours(
        image: np.ndarray,
        level: float,
        *,
        min_area: float,
) -> Contours:
    contours = find_contours(image, level)
    contours = filter_contours_by_area(contours, threshold_min=min_area)
    return sorted(contours, key=contour_area, reverse=True)
