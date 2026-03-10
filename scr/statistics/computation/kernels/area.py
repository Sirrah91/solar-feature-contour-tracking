import numpy as np

from scr.statistics.numerics.weighted import weighted_sum


def area_kernel(
        *,
        values: np.ndarray,
        weights: np.ndarray | None = None,
) -> float:
    """
    Kernel to compute area.

    values : np.ndarray
        Point density (usually np.ones_like(weights)).
    weights : np.ndarray, optional
        Weights of each value.
    """
    if weights is None:
        weights = np.ones_like(values)
    return weighted_sum(array=values, weights=weights)
