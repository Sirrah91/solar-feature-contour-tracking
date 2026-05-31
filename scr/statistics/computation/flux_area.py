import numpy as np

from scr.statistics.computation.kernels.flux import flux_kernel
from scr.statistics.computation.aggregator import aggregate_kernel
from scr.statistics.computation.masks import corr_mask, overall_mask
from scr.utils.types_alias import Mask, Masks
from scr.utils.filesystem import is_empty


def compute_flux_area(
        *,
        field2d: np.ndarray,
        masks: Masks,
        total_mask: Mask | None = None,
        projection_weights: np.ndarray | None = None,
) -> dict:
    """
    Compute flux statistics over area masks.

    Parameters
    ----------
    field2d : np.ndarray
        2D field (e.g. B).
    masks : list of Mask or None
        List of per-component masks.
    total_mask : np.ndarray | None
        Optional total mask. If not provided, computed from masks.
    projection_weights : np.ndarray or None
        Optional projection correction weights (e.g. 1/mu2d).

    Returns
    -------
    dict
        {
            "per_object": [(total, mean, std), ...] | None,
            "global": (total, mean, std) | None,
        }
    """
    if (total_mask is None) and (not is_empty(masks)):
        total_mask = overall_mask(masks)

    def _kernel(mask: np.ndarray) -> dict[str, float]:
        if is_empty(mask):
            return {"total": np.nan, "mean": np.nan, "std": np.nan}

        weights = (
            corr_mask(mask, projection_weights)
            if projection_weights is not None
            else mask
        )

        return flux_kernel(
            values=field2d,
            weights=weights,
        )

    aggregated = aggregate_kernel(
        kernel=_kernel,
        objects=masks,
        total_object=total_mask,
        listify=True,
    )

    return aggregated
