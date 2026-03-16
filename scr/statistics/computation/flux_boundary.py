import numpy as np

from scr.statistics.computation.kernels.flux import flux_kernel
from scr.statistics.computation.aggregator import aggregate_kernel
from scr.utils.filesystem import is_empty


def compute_flux_border(
        *,
        field1d: list[np.ndarray],
        projection_weights: list[np.ndarray] | None = None,
) -> dict:
    """
    Compute flux statistics along contour borders.

    Parameters
    ----------
    field1d : list[np.ndarray]
        Field sampled along each contour.
    projection_weights : list[np.ndarray] | None
        Corresponding arc-length weights (e.g. ds * 1/mu1d).

    Returns
    -------
    dict
        {
            "per_object": [(total, mean, std), ...] | None,
            "global": (total, mean, std) | None,
        }
    """

    # Normalise projection weights
    projection_weights = (
        projection_weights
        if projection_weights is not None
        else [np.ones_like(v, dtype=np.float64) for v in field1d]
    )

    # Pair values + weights into single object
    objects = list(zip(field1d, projection_weights))

    def _kernel(obj: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        values, weights = obj

        if is_empty(values):
            return {"total": np.nan, "mean": np.nan, "std": np.nan}

        return flux_kernel(
            values=values,
            weights=weights,
        )

    # Global = concatenation (union equivalent)
    if not is_empty(field1d):
        total_object = (
            np.concatenate(field1d),
            np.concatenate(projection_weights),
        )
    else:
        total_object = ([], [])

    aggregated = aggregate_kernel(
        kernel=_kernel,
        objects=objects,
        total_object=total_object,
        listify=True
    )

    return aggregated
