import numpy as np
from pwlf import PiecewiseLinFit
from matplotlib import pyplot as plt

from scr.config.figures import SAVEFIG_KWARGS
from scr.utils.naming import plural

from scr.plotting.style.fonts import font_style
from scr.plotting.generic.lines import plot_line


def plot_flux_fit_control(
        *,
        t: np.ndarray,
        total_flux: np.ndarray,
        model: PiecewiseLinFit,
        outfile: str,
        use_tex: bool = False,
) -> None:
    with font_style(fontsize=11, use_tex=use_tex):
        fig, ax = plt.subplots(figsize=(8, 6))
    
        plot_line(ax, t, total_flux,
                  line_kwargs={"color": "black", "alpha": 0.7, "label": "Raw Data", "linewidth": 1.2})
    
        t_model = model.fit_breaks
        total_flux_model = model.predict(t_model)
        plot_line(ax, t_model, total_flux_model,
                  line_kwargs={"marker": "o", "markersize": 5, "color": "red", 
                               "label": f"PWLF ({plural(len(t_model) - 1, 'segment')})", "linewidth": 2.0})
    
        ax.set_xlabel("Frame ID")
        ax.set_ylabel("Normalised Total Flux")
        ax.set_title("Flux Piecewise Linear Fit Analysis")
        
        # Aesthetic tweaks: Lighter grid, clean legend
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(frameon=True, facecolor="white", framealpha=0.9)
        
        plt.tight_layout()
        fig.savefig(outfile, **SAVEFIG_KWARGS)
        plt.close(fig)
