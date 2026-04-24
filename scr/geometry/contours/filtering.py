import numpy as np

from scr.utils.types_alias import Contour, Contours

from scr.geometry.contours.distance import contours_distance
from scr.geometry.contours.area import contour_area
from scr.geometry.contours.normalization import close_contour


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


def filter_contours_by_vertices(
        contours: Contours,
        *,
        min_vertices: int = 4,
        max_healing_gap: float = 0.0,
        max_closing_gap: float = 0.0,
) -> Contours:
    """
    Filter contours by number of vertices and optionally enforce closure.

    Parameters
    ----------
    contours : list of ndarray
        List of (N,2) arrays representing contours.
    min_vertices : int
        Minimum number of points required to keep the contour.
    max_healing_gap : float
        Maximum distance to "snap" last point to first for numerical errors.
    max_closing_gap : float
        Maximum distance to forcibly close contour by appending first point.

    Returns
    -------
    filtered : list of ndarray
        Contours that passed filtering, optionally snapped/closed.
    """
    filtered = []

    for c in contours:
        if len(c) < min_vertices:
            continue

        gap_distance = np.linalg.norm(c[0] - c[-1])

        if gap_distance == 0.0:
            # Already perfectly closed
            filtered.append(c)
        elif gap_distance <= max_healing_gap:
            # Snap small numerical gap
            c_closed = c.copy()
            c_closed[-1] = c_closed[0]
            filtered.append(c_closed)
        elif gap_distance <= max_closing_gap:
            # Append first point to forcibly close larger gap
            c_closed = close_contour(c)
            filtered.append(c_closed)
        # Otherwise: gap too large, discard contour

    return filtered
