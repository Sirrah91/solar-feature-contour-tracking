import numpy as np

from scr.utils.types_alias import Contour, Contours

from scr.geometry.contours.distance import contours_distance
from scr.geometry.contours.area import contour_area


def filter_candidate_contours(
        input_contour: Contour,
        candidates: Contours,
        max_distance: float
) -> Contours:
    """Filter candidate contours based on proximity to input."""
    return [
        cnt for cnt in candidates
        if contours_distance(input_contour, cnt) <= max_distance
    ]


def filter_contours_by_area(
        contours: Contours,
        threshold_min: float = -np.inf,
        threshold_max: float = np.inf
) -> Contours:
    """
    Filter a list of contours based on pixel area.
    Only keeps contours with area between `threshold_min` and `threshold_max`.
    """
    if threshold_min == -np.inf and threshold_max == np.inf:
        return contours
    return [c for c in contours if threshold_min <= contour_area(c) <= threshold_max]
