import numpy as np

from scr.statistics.computation.kernels.area import area_kernel
from scr.statistics.computation.aggregator import aggregate_kernel
from scr.statistics.computation.masks import overall_mask, corr_mask
from scr.utils.types_alias import Mask, Masks
from scr.utils.filesystem import is_empty


def compute_area(
        *,
        masks: Masks,
        total_mask: Mask | None = None,
        projection_weights: np.ndarray | None = None,
) -> dict:
    """
    Compute area statistics (per-object and total) using fractional masks.

    Parameters
    ----------
    masks : list[np.ndarray]
        List of masks (fractional, float 0-1).
    total_mask : np.ndarray | None
        Optional total mask. If not provided, computed from masks.
    projection_weights : np.ndarray | None
        Optional projection correction weights (e.g., 1/mu2d).

    Returns
    -------
    dict
        {"per_object": [area1, area2, ...], "global": total_area}
    """

    if (total_mask is None) and (not is_empty(masks)):
        total_mask = overall_mask(masks)

    def _kernel(mask: np.ndarray):
        if is_empty(mask):
            return np.nan

        weights = (
            corr_mask(mask, projection_weights)
            if projection_weights is not None
            else mask
        )

        return area_kernel(
            values=np.ones_like(weights, dtype=np.float64),
            weights=weights,
        )

    return aggregate_kernel(
        kernel=_kernel,
        objects=masks,
        total_object=total_mask,
        listify=False,
    )
