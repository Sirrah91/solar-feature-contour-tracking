import numpy as np

from scr.geometry.contours.fractal import fractal_dimension_contour


def fractal_kernel(
        *,
        values: np.ndarray,
        weights: np.ndarray | None = None,
) -> float:
    """
    Kernel to compute fractal dimension from a list of contours.

    values : np.ndarray
        Nx2 contour array.
    weights : np.ndarray, optional
        Optional weighting for points (usually 1s or mask values).
    """
    if weights is None:
        weights = np.ones_like(values)
    return fractal_dimension_contour(contour=values * weights, n_scales=10, control_plot=False)
