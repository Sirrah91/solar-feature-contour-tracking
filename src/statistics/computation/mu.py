import numpy as np

from src.utils.types_alias import Mask, Masks
from src.utils.filesystem import is_empty

from src.statistics.computation.masks import overall_mask
from src.statistics.computation.kernels.mu import mu_kernel
from src.statistics.computation.aggregator import aggregate_kernel


def compute_mu_stats(
        *,
        mu2d: np.ndarray,
        masks: Masks,
        total_mask: Mask | None = None,
) -> dict:
    if (total_mask is None) and (not is_empty(masks)):
        total_mask = overall_mask(masks)

    def _kernel(mask: np.ndarray) -> dict[str, float]:
        if is_empty(mask):
            return {"min": np.nan, "mean": np.nan, "max": np.nan}

        weights = np.isfinite(mask) & (mask != 0.0)
        if np.sum(weights) == 0.0:
            return {"min": np.nan, "mean": np.nan, "max": np.nan}

        return mu_kernel(values=mu2d, weights=weights)

    aggregated = aggregate_kernel(
        kernel=_kernel,
        objects=masks,
        total_object=total_mask,
        listify=True,
    )

    return aggregated
