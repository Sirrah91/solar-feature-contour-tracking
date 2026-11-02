from typing import Mapping, Sequence

import numpy as np

from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from src.config.quantities import QuantitySpec
from src.config.figures import LATEX_MATH_FONT
from src.statistics.distributions.gaussian import (
    gaussian,
    fit_gaussian_to_histogram_values,
)
from src.plotting.utils import merge_explicit_kwargs
from src.plotting.generic.lines import plot_line


QuantitySpecInput = QuantitySpec | Mapping[str, str | None] | None


def _strip_math_delimiters(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.strip()
    value = value.strip("$")

    return value or None


def _get_quantity_display_info(
        spec_quantity: QuantitySpecInput,
) -> tuple[str | None, str | None, str]:
    """Extract optional LaTeX labels and unit for quantity display."""
    if spec_quantity is None:
        return r"\mu", r"\sigma", ""

    if isinstance(spec_quantity, QuantitySpec):
        return (
            _strip_math_delimiters(spec_quantity.latex_mean),
            _strip_math_delimiters(spec_quantity.latex_std),
            spec_quantity.unit,
        )

    latex_mean = spec_quantity.get("latex_mean", r"\mu")
    latex_std = spec_quantity.get("latex_std", r"\sigma")
    unit = spec_quantity.get("unit", "")

    for name, value in (
            ("latex_mean", latex_mean),
            ("latex_std", latex_std),
            ("unit", unit),
    ):
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"Expected '{name}' to be a string or None, "
                f"got {type(value).__name__}."
            )

    return (
        _strip_math_delimiters(latex_mean),
        _strip_math_delimiters(latex_std),
        unit,
    )


def _format_gaussian_parameter(
        value: float,
        *,
        symbol: str | None,
        unit: str,
        text_prec: int,
) -> str:
    value_text = f"{value:.{text_prec}f}"

    if unit:
        value_text += rf"\,{LATEX_MATH_FONT}{{{unit}}}"

    if symbol is not None:
        value_text = rf"{symbol} = {value_text}"

    return f"${value_text}$"


def plot_hist1d(
        ax: Axes,
        datasets: Sequence[np.ndarray],
        *,
        labels: Sequence[str],
        colors: Sequence[str],
        linestyles: Sequence[str],
        bins: str | int | Sequence[float] = "auto",
        density: bool = False,
        scale: float = 1.0,
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
        counts, edges = np.histogram(d, bins=bins, density=density)
        centers = 0.5 * (edges[:-1] + edges[1:])

        h, = ax.step(
            centers,
            counts * scale,
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
        density: bool = False,
        norm: Normalize | str | None = None,
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

    return im


def overlay_gaussian_fit(
        ax: Axes,
        histogram_line: Line2D,
        *,
        spec_quantity: QuantitySpecInput = None,
        label: str = "Gaussian fit",
        line_kwargs: dict | None = None,
        text_prec: int = 0,
) -> tuple[Line2D, Line2D]:
    line_kwargs = merge_explicit_kwargs(
        line_kwargs,
        label=label,
    )

    centers, counts = histogram_line.get_data()
    popt, _ = fit_gaussian_to_histogram_values(centers, counts)

    x = np.linspace(centers.min(), centers.max(), 300)
    y = gaussian(x, *popt)

    line = plot_line(ax, x, y, line_kwargs=line_kwargs)

    mu, sigma = popt[1], popt[2]

    latex_mean, latex_std, unit = _get_quantity_display_info(spec_quantity)

    mean_text = _format_gaussian_parameter(
        mu,
        symbol=latex_mean,
        unit=unit,
        text_prec=text_prec,
    )
    std_text = _format_gaussian_parameter(
        sigma,
        symbol=latex_std,
        unit=unit,
        text_prec=text_prec,
    )

    dummy_line, = ax.plot([], [], color="none", label=f"{mean_text}, {std_text}")

    return line, dummy_line
