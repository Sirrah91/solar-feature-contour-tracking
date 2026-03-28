import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from typing import Sequence

from scr.config.quantities import MeasurementSpec

from scr.statistics.distributions.gaussian import gaussian, fit_gaussian_to_histogram

from scr.plotting.utils import merge_explicit_kwargs
from scr.plotting.generic.lines import plot_line


def plot_pdfs(
        ax: Axes,
        datasets: Sequence[np.ndarray],
        *,
        labels: Sequence[str],
        colors: Sequence[str],
        linestyles: Sequence[str],
        bins: str | int = "auto",
        pdf_kwargs: dict | None = None,
) -> list[Line2D]:
    """
    Plot 1D PDFs as stepped histograms.
    """
    pdf_kwargs = merge_explicit_kwargs(
        pdf_kwargs,
    )

    handles: list[Line2D] = []

    for d, lab, col, ls in zip(datasets, labels, colors, linestyles):
        d = d[np.isfinite(d)]
        counts, edges = np.histogram(d, bins=bins, density=True)

        h, = ax.step(
            edges[:-1],
            counts * 100,
            where="mid",
            color=col,
            linestyle=ls,
            lw=2,
            label=lab,
            **pdf_kwargs,
        )
        handles.append(h)

    return handles


def plot_hist2d(
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        *,
        bins: int | tuple[int, int] = 50,
        range: tuple[tuple[float, float], tuple[float, float]] | None = None,
        density: bool = True,
        norm: Normalize | None = None,
        cmap: str = "viridis",
        hist2d_kwargs: dict | None = None,
) -> QuadMesh:
    """
    Plot a 2D probability density histogram.
    """
    hist2d_kwargs = merge_explicit_kwargs(
        hist2d_kwargs,
        bins=bins,
        range=range,
        density=density,
        norm=norm,
        cmap=cmap,
    )

    mask = np.isfinite(x) & np.isfinite(y)

    _, _, _, im = ax.hist2d(
        x[mask],
        y[mask],
        **hist2d_kwargs,
    )

    im.set_clim(vmin=0.0)
    return im


def overlay_gaussian_fit(
        ax: Axes,
        data: np.ndarray,
        *,
        bins: str | int,
        spec: MeasurementSpec,
        label: str = "Gaussian fit",
        line_kwargs: dict | None = None,
) -> tuple[Line2D, str]:
    line_kwargs = merge_explicit_kwargs(
        line_kwargs,
        label=label,
    )

    popt, _, centers, counts = fit_gaussian_to_histogram(data, bins=bins)
    x = np.linspace(centers.min(), centers.max(), 300)
    y = gaussian(x, *popt) * 100

    line = plot_line(ax, x, y, line_kwargs=line_kwargs)

    mu, sigma = popt[1], popt[2]
    # [1:-1] to cut $
    text = rf"${spec.quantity.latex_mean[1:-1]} = {mu:.0f}\,{spec.quantity.unit}$, $\sigma={sigma:.0f}\,{spec.quantity.unit}$"

    return line, text
