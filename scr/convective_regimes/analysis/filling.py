from typing import Callable

import numpy as np

from scr.geometry.solar.units import pixelarea_to_Mm2


def analyse_penumbral_filling_vs_flux(
        Phi: list[np.ndarray],
        B: list[np.ndarray],
        gamma: list[np.ndarray],
        Ic: list[np.ndarray],
        *,
        region_function: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        ic_boundary: float = 0.9,
        n_flux_bins: int = 30,
        min_count: int = 5,
) -> dict:
    """
    Analyse penumbral-regime filling factor as a function of total magnetic flux.

    Parameters
    ----------
    Phi : per-object raw flux arrays (pixel units)
    B, gamma, Ic : per-object pixel arrays
    region_function : callable(B, gamma, Ic) → boolean mask of the target regime
    ic_boundary : upper intensity contrast threshold defining the structure boundary
    n_flux_bins : number of flux histogram bins
    min_count : minimum objects per bin to compute statistics

    Returns
    -------
    dict with keys: Phi, filling, flux_bins, flux_centers, median_filling, p16, p84
    """
    flux_scale = pixelarea_to_Mm2(px_area=1.0) * 10 ** 16
    Phi_scaled = [
        flux_scale * p[0] if len(p) > 0 else np.nan
        for p in Phi
    ]

    filling = []
    for B_i, g_i, Ic_i, Phi_i in zip(B, gamma, Ic, Phi_scaled):
        if not np.isfinite(Phi_i):
            filling.append(np.nan)
            continue

        structure = Ic_i <= ic_boundary

        if np.sum(structure) == 0:
            filling.append(np.nan)
            continue

        region = structure & region_function(B_i, g_i, Ic_i)
        filling.append(np.sum(region) / np.sum(structure))

    filling = np.asarray(filling)
    Phi_arr = np.asarray(Phi_scaled)

    valid = np.isfinite(Phi_arr) & np.isfinite(filling)
    Phi_arr = Phi_arr[valid]
    filling = filling[valid]

    flux_bins = np.linspace(np.nanmin(Phi_arr), np.nanmax(Phi_arr), n_flux_bins)
    flux_centers = flux_bins[:-1] + np.diff(flux_bins) / 2

    median_filling = np.full(len(flux_centers), np.nan)
    p16 = np.full(len(flux_centers), np.nan)
    p84 = np.full(len(flux_centers), np.nan)

    for i in range(len(flux_centers)):
        mask = (Phi_arr >= flux_bins[i]) & (Phi_arr < flux_bins[i + 1])
        if np.sum(mask) < min_count:
            continue
        vals = filling[mask]
        median_filling[i] = np.nanmedian(vals)
        p16[i] = np.nanpercentile(vals, 16, method="median_unbiased")
        p84[i] = np.nanpercentile(vals, 84, method="median_unbiased")

    return {
        "Phi": Phi_arr,
        "filling": filling,
        "flux_bins": flux_bins,
        "flux_centers": flux_centers,
        "median_filling": median_filling,
        "p16": p16,
        "p84": p84,
    }
