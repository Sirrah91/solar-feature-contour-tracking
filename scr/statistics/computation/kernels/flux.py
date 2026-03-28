import numpy as np

from scr.statistics.numerics.weighted import weighted_sum, weighted_average, weighted_std


def flux_kernel(
        *,
        values: np.ndarray,
        weights: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute total, mean, and weighted std of values.
    """
    if weights is None:
        weights = np.ones_like(values)

    total = weighted_sum(array=values, weights=weights)
    mean = weighted_average(array=values, weights=weights)
    std = weighted_std(array=values, mean=mean, weights=weights)

    return {"total": total, "mean": mean, "std": std}
