import numpy as np

from scr.utils.types_alias import Contours


def shift_contours(
        contours: Contours,
        *,
        y_offset: int,
        x_offset: int,
) -> Contours:
    """
    Shift contours by given pixel offsets.

    Parameters
    ----------
    contours : list of contours
        Each contour is an (N, 2) array of (y, x).
    y_offset, x_offset : int
        Offset to subtract (crop origin).

    Returns
    -------
    shifted_contours : list of contours
    """
    return [
        np.column_stack((c[:, 0] - y_offset, c[:, 1] - x_offset))
        for c in contours
    ]
