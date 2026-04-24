import numpy as np

from scr.utils.types_alias import Mask, Masks
from scr.statistics.computation.area import compute_area
from scr.statistics.computation.flux_area import compute_flux_area
from scr.statistics.computation.mu import compute_mu_stats


def compute_mask_geometric_stats(
        *,
        masks: Masks,
        total_mask: Mask,
        projection_weights: np.ndarray | None = None,
) -> dict:
    """
    Compute all mask-based geometry statistics for a given mask set.

    Returns
    -------
    dict
        Structured statistics dictionary.
    """

    # ------------------------
    # Area
    # ------------------------
    area_raw = compute_area(
        masks=masks,
        total_mask=total_mask,
        projection_weights=None,
    )

    area_corr = compute_area(
        masks=masks,
        total_mask=total_mask,
        projection_weights=projection_weights,
    )

    # ------------------------
    # Projection
    # ------------------------
    mu = compute_mu_stats(
        mu2d=1./projection_weights,
        masks=masks,
        total_mask=total_mask,
    )

    return {
        "area_raw": area_raw,
        "area_corr": area_corr,
        "mu": mu,
    }


def compute_mask_intensity_stats(
        *,
        masks: Masks,
        total_mask: Mask,
        image: np.ndarray,
        projection_weights: np.ndarray | None = None,
) -> dict:
    """
    Compute all mask-based flux statistics for a given mask set.

    Returns
    -------
    dict
        Structured statistics dictionary.
    """

    # ------------------------
    # Flux
    # ------------------------
    flux_raw = compute_flux_area(
        field2d=image,
        masks=masks,
        total_mask=total_mask,
        projection_weights=None,
    )

    flux_corr = compute_flux_area(
        field2d=image,
        masks=masks,
        total_mask=total_mask,
        projection_weights=projection_weights,
    )

    return {
        "flux_raw": flux_raw,
        "flux_corr": flux_corr,
    }
