import numpy as np

from scr.utils.types_alias import Contour, Contours
from scr.utils.filesystem import is_empty


def contour_signed_area(
        contour: Contour,
) -> float:
    """
    Compute the signed area of a contour with optional per-point correction.

    Parameters
    ----------
    contour : (N,2) array
        Ordered coordinates (row=y, col=x).

    Returns
    -------
    float
        Signed area, corrected.
    """

    x, y = contour[:, 1], contour[:, 0]

    # To avoid overflow
    x, y = x.astype(float), y.astype(float)
    x -= x.mean()
    y -= y.mean()

    # shoelace formula with correction applied per segment
    area = 0.5 * (np.dot(x, np.roll(y, shift=1)) - np.dot(y, np.roll(x, shift=1)))
    return area


def contour_area(
        contour: Contour,
) -> float:
    return np.abs(contour_signed_area(contour))


def total_contours_area(
        contours: Contours,
        hole_contours: Contours | None = None
) -> float:
    """
    Estimate the total area enclosed by a list of 2D contours using the shoelace formula.

    Parameters:
        contours: List of Nx2 arrays representing ordered (y, x) or (row, col) coordinates of polygons.
        hole_contours: List of inner hole contours.

    Returns:
        Total area of all contours combined.
    """
    if not is_empty(hole_contours):
        return sum(contour_signed_area(contour) for contour in contours) - sum(
            contour_signed_area(contour) for contour in hole_contours)

    return sum(contour_signed_area(contour) for contour in contours)
