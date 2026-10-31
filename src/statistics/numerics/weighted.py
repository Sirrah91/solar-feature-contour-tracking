import numpy as np


def weighted_sum(
        *,
        array: np.ndarray,
        weights: np.ndarray,
) -> float:
    mask = np.isfinite(array) & np.isfinite(weights)
    if not np.any(mask):
        return np.nan

    w = weights[mask]
    w_sum = np.sum(w)
    if w_sum == 0.0:
        return np.nan

    return float(np.sum(array[mask] * w))


def weighted_average(
        *,
        array: np.ndarray,
        weights: np.ndarray,
) -> float:
    mask = np.isfinite(array) & np.isfinite(weights)
    if not np.any(mask):
        return np.nan

    w = weights[mask]
    w_sum = np.sum(w)
    if w_sum == 0.0:
        return np.nan

    return float(np.sum(array[mask] * w) / w_sum)
    # return float(np.average(array[mask], weights=w))


def weighted_std(
        *,
        array: np.ndarray,
        mean: float,
        weights: np.ndarray,
) -> float:
    if not np.isfinite(mean):
        return np.nan

    mask = np.isfinite(array) & np.isfinite(weights)
    if not np.any(mask):
        return np.nan

    w = weights[mask]
    w_sum = np.sum(w)
    if w_sum == 0.0:
        return np.nan

    variance = np.sum(w * (array[mask] - mean) ** 2.0) / w_sum
    return float(np.sqrt(variance))
    # return np.sqrt(weighted_average(array=(array[mask] - mean) ** 2., weights=w))
