import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from os import path

from src.config.paths import PATH_SUNSPOTS_PHASES, PATH_FIGURES
from src.config.quantities import get_measurement_spec
from src.config.figures import SAVEFIG_KWARGS, FIG_FORMAT

from src.utils.types_alias import Quantity, SunspotPart
from src.utils.filesystem import mkdir
from src.io.parquet import load_parquet
from src.statistics.aggregations import lifetime_and_mean

from src.plotting.generic.scatter import plot_scatter
from src.plotting.style.fonts import font_style


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
    mkdir(figure_outdir)

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

        ax.set_xlabel(r"Lifetime\,(h)")
        ax.set_ylabel(spec.label_mean())

        plt.tight_layout()

        fig.savefig(
            path.join(figure_outdir, f"lifetime_{QUANTITY}_{WHERE}.{FIG_FORMAT}"),
            format=FIG_FORMAT,
            **SAVEFIG_KWARGS,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
