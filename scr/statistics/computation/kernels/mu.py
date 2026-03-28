import numpy as np


def mu_kernel(
        *,
        values: np.ndarray,
        weights: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute mean, min, and max of values.
    """
    if weights is None:
        weights = np.ones_like(values, dtype=bool)

    _min = np.nanmin(values[weights])
    mean = np.nanmean(values[weights])
    _max = np.nanmax(values[weights])

    return {"min": _min, "mean": mean, "max": _max}
