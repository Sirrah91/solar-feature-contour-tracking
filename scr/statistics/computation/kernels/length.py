import numpy as np

from scr.geometry.contours.length import contour_length


def length_kernel(
        *,
        values: np.ndarray,
        weights: np.ndarray | None = None,
) -> float:
    """
    Kernel to compute contour lengths.

    Parameters
    ----------
    values : np.ndarray
        Nx2 array of contour points (lon, lat).
    weights : np.ndarray, optional
        Weights of each value.

    Returns
    -------
    float
    """
    if weights is None:
        weights = np.ones_like(values)
    return contour_length(values * weights)
