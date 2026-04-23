import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from os import path

from scr.config.paths import PATH_SUNSPOTS_PHASES, PATH_FIGURES
from scr.config.filtering import gimme_filtering_kwargs
from scr.config.quantities import get_measurement_spec
from scr.config.figures import FIG_FORMAT, SAVEFIG_KWARGS, CBAR_KWARGS

from scr.utils.types_alias import Quantity, SunspotPart, ObjectFilteringMode
from scr.utils.filesystem import check_dir
from scr.geometry.solar.units import pixelarea_to_Mm2
from scr.io.parquet import load_parquet
from scr.statistics.dataframe.filtering import filter_dataset_by_spots

from scr.plotting.generic.hist import plot_hist2d
from scr.plotting.style.fonts import font_style
from scr.plotting.style.colorbar import add_colorbar


def main():
    """
    Plot a 2D PDF of magnetic-field statistics versus region area
    for sunspots or pores.
    """
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    QUANTITY: Quantity = "B"
    WHERE: SunspotPart = "sunspot"
    MODE: ObjectFilteringMode = "sunspots"
    PATH_SUNSPOTS_PHASES = "/nfsscratch/david/Contours/sunspots_removed_phases"

    spec = get_measurement_spec(QUANTITY, WHERE)

    figure_outdir = path.join(PATH_FIGURES, "paper_plots")
    check_dir(figure_outdir)

    # ------------------------------------------------------------------
    # Load and filter combined phase data
    # ------------------------------------------------------------------
    df = load_parquet(
        path.join(PATH_SUNSPOTS_PHASES, f"sunspots_phases.parquet"),
        return_dataset=True
    )
    df = filter_dataset_by_spots(df, gimme_filtering_kwargs(MODE))
    df = df[["sunspot_area_corr", spec.mean_col, spec.std_col]]

    # ------------------------------------------------------------------
    # Prepare quantities for plotting
    # ------------------------------------------------------------------
    # Convert total area from pixel units to Mm^2
    area = pixelarea_to_Mm2(df["sunspot_area_corr"].to_numpy())

    # Quality cut on field dispersion
    std_threshold = 130.0 if MODE == "sunspots" else 300.0
    good = df[spec.std_col] < std_threshold

    mean = df.loc[good, spec.mean_col].to_numpy()
    area = area[good]

    del df  # free memory early

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    with font_style(fontsize=20):
        fig, ax = plt.subplots(figsize=(10, 10))

        im = plot_hist2d(
            ax,
            mean,
            area,
            cmap="cubehelix_r",
        )

        add_colorbar(
            ax,
            im,
            cbar_kwargs=CBAR_KWARGS,
            label=r"PDF (\%)",
            formatter=FuncFormatter(lambda v, _: f"{100 * v:.3f}"),
        )

        ax.set_xlabel(spec.ylabel_mean)
        ax.set_ylabel(
            fr"{'Sunspot' if MODE == 'sunspots' else 'Pore'} area (Mm$^2$)"
        )

        plt.tight_layout()

        fig.savefig(
            path.join(
                figure_outdir,
                f"area_{QUANTITY}_{MODE}_{WHERE}.{FIG_FORMAT}",
            ),
            format=FIG_FORMAT,
            **SAVEFIG_KWARGS,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
