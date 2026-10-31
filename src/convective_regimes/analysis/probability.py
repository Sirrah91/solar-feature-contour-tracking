import numpy as np

from src.convective_regimes.core.models import ProbabilityMap2D


def analyse_penumbra_probability(
        B: np.ndarray,
        gamma: np.ndarray,
        Ic: np.ndarray,
        *,
        ic_penumbra: tuple[float, float] = (0.5, 0.9),
        ic_reference: tuple[float, float] | None = None,
        min_count: int = 50,
        n_B_bins: int = 100,
        n_g_bins: int = 45,
) -> ProbabilityMap2D:
    """
    Compute 2D conditional penumbra probability in (B, gamma) space.

    Parameters
    ----------
    B, gamma, Ic : pixel arrays (ravelled internally)
    ic_penumbra : (lower, upper) intensity contrast bounds defining penumbra
    min_count : bins with fewer total counts are masked to NaN
    n_B_bins, n_g_bins : histogram resolution

    Returns
    -------
    ProbabilityMap2D with x_bins=B_bins, y_bins=gamma_bins
    """
    B = np.asarray(B).ravel()
    gamma = np.asarray(gamma).ravel()
    Ic = np.asarray(Ic).ravel()

    target = (
            (ic_penumbra[0] <= Ic) & (Ic <= ic_penumbra[1])
    ).astype(int)

    valid = (
            np.isfinite(B)
            & np.isfinite(gamma)
            & np.isfinite(Ic)
            & (B <= 5000.0)
    )
    if ic_reference is not None:
        valid &= (ic_reference[0] <= Ic) & (Ic <= ic_reference[1])

    B = B[valid]
    gamma = gamma[valid]
    target = target[valid]

    B_bins = np.linspace(0, np.nanmax(B), n_B_bins)
    g_bins = np.linspace(0, 90, n_g_bins)

    H2_total, _, _ = np.histogram2d(B, gamma, bins=[B_bins, g_bins])
    H2_target, _, _ = np.histogram2d(
        B[target == 1], gamma[target == 1], bins=[B_bins, g_bins],
    )

    probability = H2_target / (H2_total + 1e-9)
    probability[H2_total < min_count] = np.nan

    return ProbabilityMap2D(
        probability=probability,
        counts=H2_target,
        x_bins=B_bins,
        y_bins=g_bins,
    )
