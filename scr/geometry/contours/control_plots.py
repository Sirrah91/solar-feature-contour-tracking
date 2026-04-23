import numpy as np
from matplotlib import pyplot as plt

from scr.config.figures import SAVEFIG_KWARGS

from scr.plotting.style.fonts import font_style
from scr.plotting.generic.lines import plot_line


def plot_fractal_dimension_control(
        *,
        logs_eps: np.ndarray,
        logs_N: np.ndarray,
        logs_N_fit: np.ndarray,
        fractal_dim: float,
        outfile: str,
        use_tex: bool = False,
) -> None:
    with font_style(fontsize=11, use_tex=use_tex):
        fig, ax = plt.subplots(figsize=(8, 6))

        plot_line(ax, logs_eps, logs_N,
                  line_kwargs={"marker": "o", "markersize": 4, "linestyle": ":",
                               "color": "black", "alpha": 0.6, "label": "Observed Counts"})
        plot_line(ax, logs_eps, logs_N_fit,
                  line_kwargs={"color": "red", "linewidth": 2,
                               "label": rf"Linear Fit ($D \approx {fractal_dim:.3f}$)"})

        ax.set_xlabel(r"$\log(1 / \mathrm{box\ size})$")
        ax.set_ylabel(r"$\log(N_\mathrm{boxes})$")
        ax.set_title("Box-Counting Dimension Diagnostic")

        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(frameon=True, facecolor="white", framealpha=0.9)

        plt.tight_layout()
        fig.savefig(outfile, **SAVEFIG_KWARGS)
        plt.close(fig)
