import numpy as np

from scr.statistics.computation.kernels.length import length_kernel
from scr.statistics.computation.aggregator import aggregate_kernel
from scr.utils.types_alias import Contours
from scr.utils.filesystem import is_empty


def _sum_or_nan(values: list[float]) -> float:
    """Internal helper to sum a list but return NaN if empty."""
    if is_empty(values):
        return np.nan

    return np.nansum(values)


def compute_lengths(
        *,
        contours: Contours,
        total_length: bool = False,
        projection_weights: np.ndarray | None = None,
) -> dict:
    def _kernel(contour: np.ndarray):
        if is_empty(contour):
            return np.nan

        return length_kernel(
            values=contour,
            weights=projection_weights,
        )

    return aggregate_kernel(
        kernel=_kernel,
        objects=contours,
        total_object=total_length,
        listify=False,
        reduce_global=_sum_or_nan,
    )


def compute_spherical_lengths(
        *,
        arc_lengths: list[np.ndarray],
        total_length: bool = False,
        projection_weights: np.ndarray | None = None,
) -> dict:
    if projection_weights is None:
        projection_weights = np.float64(1.0)

    def _kernel(arc_length: np.ndarray):
        if is_empty(arc_length):
            return np.nan

        return np.nansum(
            arc_length * projection_weights
        )

    return aggregate_kernel(
        kernel=_kernel,
        objects=arc_lengths,
        total_object=total_length,
        listify=False,
        reduce_global=_sum_or_nan,
    )
