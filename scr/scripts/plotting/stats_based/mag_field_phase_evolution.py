import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from os import path

from scr.config.paths import PATH_SUNSPOTS_PHASES, PATH_FIGURES
from scr.config.filtering import gimme_filtering_kwargs
from scr.config.quantities import get_measurement_spec
from scr.config.figures import FIG_FORMAT, SAVEFIG_KWARGS

from scr.utils.types_alias import Quantity, SunspotPart, ObjectFilteringMode
from scr.utils.filesystem import check_dir
from scr.io.parquet import load_parquet
from scr.statistics.dataframe.filtering import filter_dataset_by_spots
from scr.statistics.segments.phase import extract_phase_segments, median_curve

from scr.plotting.generic.lines import plot_line
from scr.plotting.composite.phase_segments import plot_phase_segments
from scr.plotting.style.fonts import font_style


def main():
    """
    Plot temporal evolution of a quantity across evolutionary phases
    (forming → stable → decaying), including median trends.
    """
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    QUANTITY: Quantity = "Bhor"
    WHERE: SunspotPart = "Ic<0.5"
    MODE: ObjectFilteringMode = "pores"
    PATH_SUNSPOTS_PHASES = "/nfsscratch/david/Contours/sunspots_removed_phases"

    spec = get_measurement_spec(QUANTITY, WHERE)

    phases = ("forming", "stable", "decaying")

    figure_outdir = path.join(PATH_FIGURES, "paper_plots")
    check_dir(figure_outdir)

    # ------------------------------------------------------------------
    # Load and filter data
    # ------------------------------------------------------------------
    df = load_parquet(
        path.join(PATH_SUNSPOTS_PHASES, f"sunspots_phases.parquet"),
        return_dataset=True
    )
    df = filter_dataset_by_spots(df, gimme_filtering_kwargs(MODE))
    df = df[[spec.mean_col, "spot_global_index", "frame", "phase"]]

    # Extract time-normalised segments for each phase
    segments = {
        phase: extract_phase_segments(df, phase, spec.mean_col)
        for phase in phases
    }

    del df  # free memory

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    with font_style(fontsize=20):
        fig, ax = plt.subplots(figsize=(10, 10))

        for i, phase in enumerate(phases):
            # Individual phase segments
            plot_phase_segments(ax, segments[phase], i)

            # Median evolution curve overlaid
            plot_line(
                ax,
                segments[phase][0][0] + i,
                median_curve(segments[phase]),
                line_kwargs={"color": "black", "linewidth": 2.0},
            )

        # Phase-labelled x-axis
        ax.set_xlim(0, len(phases))
        ax.set_xticks([i + 0.5 for i in range(len(phases))])
        ax.set_xticklabels([p.title() for p in phases])

        ax.set_ylabel(spec.label_mean())

        plt.tight_layout()

        fig.savefig(
            path.join(figure_outdir, f"phase_evolution_{QUANTITY}_{MODE}_{WHERE}.{FIG_FORMAT}"),
            format=FIG_FORMAT,
            **SAVEFIG_KWARGS,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
