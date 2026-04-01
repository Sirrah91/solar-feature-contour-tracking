import numpy as np

from scr.utils.types_alias import Contour, Contours


def normalize_contour_input(
        contour: Contour | Contours
) -> Contours:
    """
    Ensure contour input is returned as a list of (N, 2) arrays.

    Parameters:
        contour: Either a single contour (shape (N, 2)) or a list of such contours.

    Returns:
        A list of contours, where each contour is a NumPy array of shape (N, 2).

    Notes:
        This is useful for normalising input before passing to functions
        that expect a list of contours, while allowing flexible input.
    """
    if isinstance(contour, np.ndarray):
        return [close_contour(contour)]
    else:
        return [close_contour(c) for c in contour]


def close_contour(
        contour: Contour
) -> Contour:
    """Ensure contour is closed by appending first point if needed."""
    if len(contour) > 1 and not np.array_equal(contour[0], contour[-1]):
        return np.vstack([contour, contour[0]])
    return contour
