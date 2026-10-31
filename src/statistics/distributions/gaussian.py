import numpy as np
from scipy.optimize import curve_fit


def gaussian(
        x: np.ndarray,
        A: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
) -> np.ndarray:
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def fit_gaussian_to_histogram(
        data: np.ndarray,
        *,
        bins: str | int = "auto"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts, edges = np.histogram(data[np.isfinite(data)], bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    p0 = [counts.max(), centers[np.argmax(counts)], 50]
    popt, pcov = curve_fit(gaussian, centers, counts, p0=p0)

    return popt, pcov, centers, counts
