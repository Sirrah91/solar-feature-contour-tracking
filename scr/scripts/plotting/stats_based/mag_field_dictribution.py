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
from scr.statistics.dataframe.filtering import filter_dataset_by_spots, filter_combined_df

from scr.plotting.generic.hist import plot_pdfs, overlay_gaussian_fit
from scr.plotting.style.fonts import font_style


def main():
    """
    Plot PDFs of magnetic-field statistics for different evolutionary phases
    and overlay a Gaussian fit to a quality-selected stable subset.
    """
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    QUANTITY: Quantity = "Bhor"
    WHERE: SunspotPart = "Ic<0.5"
    MODE: ObjectFilteringMode = "pores"
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

    # ------------------------------------------------------------------
    # Extract phase-wise distributions
    # ------------------------------------------------------------------
    forming = filter_combined_df(
        df,
        filtering_kwargs={
            "phase": {"exact_value": "forming", "mode": "frame-wise"}
        },
    )[spec.mean_col].to_numpy()

    stable_df = filter_combined_df(
        df,
        filtering_kwargs={
            "phase": {"exact_value": "stable", "mode": "frame-wise"}
        },
    )[[spec.mean_col, spec.std_col]]  # keep both mean & dispersion

    decaying = filter_combined_df(
        df,
        filtering_kwargs={
            "phase": {"exact_value": "decaying", "mode": "frame-wise"}
        },
    )[spec.mean_col].to_numpy()

    del df  # free memory early

    # ------------------------------------------------------------------
    # Quality cut for Gaussian fit
    # ------------------------------------------------------------------
    std_threshold = 130.0 if MODE == "sunspots" else 300.0
    good = stable_df[spec.std_col] < std_threshold

    stable = stable_df[spec.mean_col].to_numpy()
    stable_subset = stable_df.loc[good, spec.mean_col].to_numpy()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    with font_style(fontsize=20):
        fig, ax = plt.subplots(figsize=(10, 10))

        handles = plot_pdfs(
            ax,
            [forming, stable, decaying, stable_subset],
            labels=["Forming", "Stable", "Decaying", "Stable selection"],
            colors=["#1f77b4", "#2ca02c", "#ff7f0e", "#2ca02c"],
            linestyles=["-", "-", "-", "--"],
        )

        # Gaussian fit to the quality-selected stable subset
        fit_line, fit_label = overlay_gaussian_fit(
            ax,
            stable_subset,
            bins="auto",
            spec=spec,
            line_kwargs={
                "color": "black",
                "linewidth": 2.0,
            }
        )
        handles.append(fit_line)

        # Dummy handle to show fit parameters in the legend
        dummy_line = ax.plot([], [], color="none", label=fit_label)
        handles.append(dummy_line)

        ax.legend()

        ax.set_xlabel(spec.ylabel_mean)
        ax.set_ylabel(r"PDF (\%)")

        ax.set_xlim((400, 1200))
        plt.tight_layout()

        fig.savefig(
            path.join(
                figure_outdir,
                f"distribution_{QUANTITY}_{MODE}_{WHERE}.{FIG_FORMAT}",
            ),
            format=FIG_FORMAT,
            **SAVEFIG_KWARGS,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
