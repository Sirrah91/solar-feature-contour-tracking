import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
from os import path

from scr.config.paths import PATH_SUNSPOTS_PHASES, PATH_FIGURES
from scr.config.quantities import get_measurement_spec
from scr.config.figures import SAVEFIG_KWARGS, FIG_FORMAT

from scr.utils.types_alias import Quantity, SunspotPart, SunspotPhase
from scr.utils.filesystem import check_dir
from scr.io.parquet import load_parquet
from scr.statistics.aggregations import phase_duration_statistics
from scr.statistics.segments.simple_pwlf import piecewise_linear_fit

from scr.plotting.generic.lines import plot_line
from scr.plotting.style.fonts import font_style


def main():
    """
    Analyse how characteristic field strengths scale with phase duration.
    The example shown here focuses on a single evolutionary phase.
    """
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    QUANTITY: Quantity = "B"
    WHERE: SunspotPart = "sunspot"
    PHASE: SunspotPhase = "stable"
    PATH_SUNSPOTS_PHASES = "/nfsscratch/david/Contours/sunspots_removed_phases"

    spec = get_measurement_spec(QUANTITY, WHERE)

    figure_outdir = path.join(PATH_FIGURES, "paper_plots")
    check_dir(figure_outdir)

    # ------------------------------------------------------------------
    # Load and filter data
    # ------------------------------------------------------------------
    df = load_parquet(
        path.join(PATH_SUNSPOTS_PHASES, "sunspots_phases.parquet"),
        return_dataset=True
    )

    # Quality and phase selection
    arrow_filter = (
            (pc.field(spec.std_col) < 500.0) &
            (pc.field("phase") == PHASE)
    )
    df = df.to_table(
        filter=arrow_filter,
        columns=["spot_global_index", spec.mean_col, "phase_duration"],
    ).to_pandas()

    # Aggregate statistics as a function of phase duration
    stats = phase_duration_statistics(
        df,
        value_col=spec.mean_col,
    )

    del df  # free memory

    # ------------------------------------------------------------------
    # Prepare arrays
    # ------------------------------------------------------------------
    mask = np.isfinite(stats["max"])
    x = stats["duration"][mask]
    y_max = stats["max"][mask]
    y_p98 = stats["p98"][mask]
    y_p95 = stats["p95"][mask]

    # Sort by duration for plotting and fitting
    order = np.argsort(x)
    x = x[order]
    y_max = y_max[order]
    y_p98 = y_p98[order]
    y_p95 = y_p95[order]

    # Piecewise-linear fit to the maximum values
    model = piecewise_linear_fit(x, y_max, n_segments=2)
    x_model = model.fit_breaks
    y_max_model = model.predict(x_model)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    with font_style(fontsize=20):
        fig, ax = plt.subplots(figsize=(8, 6))

        plot_line(
            ax, x, y_max,
            line_kwargs={"color": "red", "linestyle": "dotted", "label": "Max"},
        )
        plot_line(
            ax, x, y_p98,
            line_kwargs={"color": "blue", "linestyle": "dotted", "label": "Percentile 98"},
        )
        plot_line(
            ax, x, y_p95,
            line_kwargs={"color": "green", "linestyle": "dotted", "label": "Percentile 95"},
        )
        plot_line(
            ax, x_model, y_max_model,
            line_kwargs={"marker": "o", "color": "black", "linestyle": "solid", "label": "PWLF"},
        )

        ax.set_xlabel(r"Phase duration\,(h)")
        ax.set_ylabel(spec.label_mean())
        ax.legend()

        plt.tight_layout()

        fig.savefig(
            path.join(figure_outdir, f"lifetime_{PHASE}_{QUANTITY}_{WHERE}.{FIG_FORMAT}"),
            format=FIG_FORMAT,
            **SAVEFIG_KWARGS,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
