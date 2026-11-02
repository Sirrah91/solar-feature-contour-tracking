from typing import Sequence
import numpy as np
from scipy.optimize import curve_fit


def gaussian(
        x: np.ndarray,
        A: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
) -> np.ndarray:
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def fit_gaussian_to_histogram_values(
        centers: np.ndarray,
        counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers, dtype=float)
    counts = np.asarray(counts, dtype=float)

    if centers.shape != counts.shape or centers.ndim != 1:
        raise ValueError("Centers and counts must be 1D arrays of equal shape.")

    mask = np.isfinite(centers) & np.isfinite(counts)
    centers = centers[mask]
    counts = counts[mask]

    if centers.size < 3:
        raise ValueError("At least three valid histogram bins are required.")

    peak_idx = np.argmax(counts)
    weights = np.maximum(counts, 0)
    mu0 = np.average(centers, weights=weights)
    sigma0 = np.sqrt(np.average((centers - mu0) ** 2, weights=weights))

    sigma0 = max(sigma0, np.min(np.diff(centers)))
    p0 = [counts[peak_idx], centers[peak_idx], sigma0]

    popt, pcov = curve_fit(
        gaussian,
        centers,
        counts,
        p0=p0,
        bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]),
    )

    return popt, pcov


def fit_gaussian_to_histogram(
        data: np.ndarray,
        *,
        bins: str | int | Sequence[float] = "auto",
        density: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts, edges = np.histogram(data[np.isfinite(data)], bins=bins, density=density)
    centers = (edges[:-1] + edges[1:]) / 2
    
    popt, pcov = fit_gaussian_to_histogram_values(centers=centers, counts=counts)

    return popt, pcov, centers, counts
