import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from os import path

from scr.config.paths import PATH_SUNSPOTS_PHASES, PATH_FIGURES
from scr.config.quantities import get_measurement_spec
from scr.config.figures import SAVEFIG_KWARGS, FIG_FORMAT

from scr.utils.types_alias import Quantity, SunspotPart
from scr.utils.filesystem import check_dir
from scr.io.parquet import load_parquet
from scr.statistics.aggregations import lifetime_and_mean

from scr.plotting.generic.scatter import plot_scatter
from scr.plotting.style.fonts import font_style


def main():
    """
    Scatter plot of mean magnetic quantity versus sunspot lifetime.
    """
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    QUANTITY: Quantity = "B"
    WHERE: SunspotPart = "sunspot"
    PATH_SUNSPOTS_PHASES = "/nfsscratch/david/Contours/sunspots_removed_phases"

    spec = get_measurement_spec(QUANTITY, WHERE)

    figure_outdir = path.join(PATH_FIGURES, "paper_plots")
    check_dir(figure_outdir)

    # ------------------------------------------------------------------
    # Load data and compute aggregations
    # ------------------------------------------------------------------
    df = load_parquet(
        path.join(PATH_SUNSPOTS_PHASES, "sunspots_phases.parquet"),
        return_dataset=True,
    )

    columns = ["spot_global_index", spec.mean_col]
    df = df.to_table(columns=columns).to_pandas()

    # Lifetime (hours) and corresponding mean value per object
    lifetime, mean_values = lifetime_and_mean(
        df,
        value_col=spec.mean_col,
    )

    del df  # free memory

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    with font_style(fontsize=20):
        fig, ax = plt.subplots(figsize=(8, 6))

        plot_scatter(
            ax,
            lifetime,
            mean_values,
        )

        ax.set_xlabel("Lifetime (h)")
        ax.set_ylabel(spec.ylabel_mean)

        plt.tight_layout()

        fig.savefig(
            path.join(figure_outdir, f"lifetime_{QUANTITY}_{WHERE}.{FIG_FORMAT}"),
            format=FIG_FORMAT,
            **SAVEFIG_KWARGS,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
