from os import path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.ticker import FuncFormatter

from scr.config.figures import CBAR_KWARGS, SAVEFIG_KWARGS
from scr.config.filtering import gimme_filtering_kwargs
from scr.config.quantities import _QUANTITIES
from scr.convective_regimes.core.regions import (
    SUNSPOT_PQ_B_THRESHOLD,
    TRANSITION_GAMMA_MIN,
    TRANSITION_GAMMA_MAX,
)
from scr.convective_regimes.core.thresholds import RegressionThresholds, compute_thresholds
from scr.convective_regimes.io.loaders import extract_q, load_all
from scr.convective_regimes.settings import (
    DATA_DIR, FIG_FORMAT, FIGURE_DIR, SUNSPOTS_PHASES_DIR_TEMPLATE,
)
from scr.convective_regimes.utils.types_alias import SunspotPhase
from scr.io.parquet import load_parquet
from scr.plotting.generic.hist import plot_hist2d, plot_pdfs
from scr.plotting.style.lines import LineSpec
from scr.plotting.style.colorbar import add_colorbar
from scr.plotting.style.fonts import font_style
from scr.statistics.dataframe.filtering import filter_combined_df, filter_dataset_by_spots
from scr.utils.filesystem import check_dir


def _distribution_properties(data: np.ndarray) -> None:
    """Print mean and FWHM of a distribution via histogram interpolation."""
    counts, bin_edges = np.histogram(data, bins="auto")
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mean_binned = np.average(bin_centers, weights=counts)

    max_index = np.argmax(counts)
    half_max = counts[max_index] / 2.0

    x_left = np.interp(half_max, counts[: max_index + 1], bin_centers[: max_index + 1])
    x_right = np.interp(
        half_max,
        counts[max_index:][::-1],
        bin_centers[max_index:][::-1],
    )
    fwhm = x_right - x_left

    print(f"  Mean: {mean_binned:.1f}    FWHM: {fwhm:.1f}")


def _build_reference_lines(t: RegressionThresholds) -> dict[str, list[LineSpec]]:
    """
    Build per-quantity reference line specs from computed thresholds.

    Missing (invalid) estimates are silently dropped.
    Geometry constants from regions.py need no guard.
    """
    return {
        "B": [
            LineSpec(SUNSPOT_PQ_B_THRESHOLD, "magenta", "-"),
        ],
        "Bhor": [
            s for s, e in (
                (LineSpec(t.sunspot_pq_bhor.mean, "blue", "-"), t.sunspot_pq_bhor),
            )
            if e.is_valid
        ],
        "Br": [LineSpec(1000.0, "red", "--")] + [
            s for s, e in (
                (LineSpec(t.sunspot_up_bver.mean, "red", "-"),  t.sunspot_up_bver),
            )
            if e.is_valid
        ],
        "Binc": [
            LineSpec(TRANSITION_GAMMA_MIN, "purple", "-"),
            LineSpec(TRANSITION_GAMMA_MAX, "purple", "--"),
        ],
    }


def plot_2d_histograms(
        phase: SunspotPhase = "all",
        *,
        data_dir: str = DATA_DIR,
        figure_outdir: str = FIGURE_DIR,
        fig_format: str = FIG_FORMAT,
        bins: int = 100,
) -> None:
    """Plot 2D (Ic, B/Br/Bhor/Binc) histograms for sunspots with pore contours overlaid."""
    check_dir(figure_outdir)

    thresholds = compute_thresholds(phase, data_dir)
    reference_lines = _build_reference_lines(thresholds)

    data_sunspots = load_all(["sunspots"], [phase], data_dir=data_dir)
    data_pores = load_all(["pores"], [phase], data_dir=data_dir)

    Ic_s = extract_q(data_sunspots, "Ic")
    Ic_p = extract_q(data_pores, "Ic")

    xlim = (0.0, 1.3)

    for q in ("B", "Br", "Bhor", "Binc"):
        y_s = extract_q(data_sunspots, q)
        y_p = extract_q(data_pores, q)
        ylim = (0.0, 3000.0) if q != "Binc" else (0.0, 90.0)

        H_p, xedges, yedges = np.histogram2d(
            Ic_p, y_p, bins=bins, range=(xlim, ylim)
        )
        H_p = H_p / H_p.sum()
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xc, yc, indexing="ij")

        mask = H_p > 0
        levels = np.quantile(
            H_p[mask],
            list(np.arange(0.6, 1.0, 0.05)) + [0.97, 0.99],
            method="median_unbiased",
        )

        with font_style(fontsize=16):
            fig, ax = plt.subplots(figsize=(8, 6))

            im = plot_hist2d(
                ax, x=Ic_s, y=y_s, bins=bins,
                cmap="cubehelix_r",
                range=(xlim, ylim),
                norm=PowerNorm(gamma=0.5, vmin=0.0),
            )
            ax.contour(X, Y, H_p, levels=levels, colors="white", linewidths=0.5)

            add_colorbar(
                ax, im, cbar_kwargs=CBAR_KWARGS,
                label=r"PDF (\%)",
                formatter=FuncFormatter(lambda v, _: f"{100 * v:.3f}"),
            )

            ax.set_title("Sunspots and pores contours")
            ax.set_xlabel(_QUANTITIES["Ic"].latex)
            try:
                Q = _QUANTITIES[q]
                ax.set_ylabel(f"{Q.latex} ({Q.unit})")
            except KeyError:
                ax.set_ylabel(rf"$\gamma$ (deg)")

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            for spec in reference_lines.get(q, []):
                ax.axhline(
                    y=spec.level,
                    color=spec.color,
                    linestyle=spec.linestyle,
                    linewidth=spec.linewidth,
                )

            ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1)
            ax.axvline(x=0.9, color="black", linestyle="--", linewidth=1)

            plt.tight_layout()
            fig.savefig(
                path.join(
                    figure_outdir,
                    f"{q}_sunspots_with_pores_contours_{phase}.{fig_format}",
                ),
                format=fig_format,
                **SAVEFIG_KWARGS,
            )
            plt.close(fig)


def plot_1d_histograms(
        *,
        sunspot_type: str = "removed",
        sunspots_phases_dir: str | None = None,
        figure_outdir: str = FIGURE_DIR,
        fig_format: str = FIG_FORMAT,
) -> None:
    """Plot 1D Bhor PDFs across phases and intensity regions."""
    check_dir(figure_outdir)

    if sunspots_phases_dir is None:
        sunspots_phases_dir = SUNSPOTS_PHASES_DIR_TEMPLATE.format(
            sunspot_type=sunspot_type
        )

    df = load_parquet(
        path.join(sunspots_phases_dir, "sunspots_phases.parquet"),
        return_dataset=True,
    )
    pores_df = filter_dataset_by_spots(df, gimme_filtering_kwargs("pores"))
    df_sunspots = filter_dataset_by_spots(df, gimme_filtering_kwargs("sunspots"))

    regions = ["Ic<0.5", "Ic<0.65", "Ic<0.9"]
    region_labels = [
        r"$I^\mathrm{c} = 0.5I^\mathrm{c}_\mathrm{QS}$",
        r"$I^\mathrm{c} = 0.65I^\mathrm{c}_\mathrm{QS}$",
        r"$I^\mathrm{c} = 0.9I^\mathrm{c}_\mathrm{QS}$",
    ]
    col = "Bhor_{reg}_flux-border_corr_mean"

    _phase_filter = lambda p: filter_combined_df(
        df_sunspots,
        filtering_kwargs={"phase": {"exact_value": p, "mode": "frame-wise"}},
    )

    datasets = [
        [
            pores_df[col.format(reg=reg)],
            _phase_filter("forming")[col.format(reg=reg)],
            _phase_filter("stable")[col.format(reg=reg)],
            _phase_filter("decaying")[col.format(reg=reg)],
        ]
        for reg in regions
    ]

    with font_style(fontsize=16):
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(19, 6))

        for ax, dataset, region_label in zip(axes, datasets, region_labels):
            plot_pdfs(
                ax, dataset,
                labels=["Pores", "Forming", "Stable", "Decaying"],
                colors=["red", "blue", "green", "black"],
                linestyles=["-", "-", "-", "-"],
            )
            ax.legend(loc="upper right")
            ax.set_title(region_label)
            ax.set_xlabel(r"$B_{\mathrm{hor}} \, \left( \mathrm{G} \right)$")
            ax.set_xlim((300, 1700))

        axes[0].set_ylabel(r"$\mathrm{PDF} \, \left( \% \right)$")
        plt.tight_layout()

        fig.savefig(
            path.join(figure_outdir, f"distribution_Bhor.{fig_format}"),
            format=fig_format,
            **SAVEFIG_KWARGS,
        )
        plt.close(fig)

    print("Distribution properties:")
    for dataset, reg in zip(datasets, regions):
        for data, label in zip(dataset, ["Pores", "Forming", "Stable", "Decaying"]):
            print(f"  {reg} / {label}")
            _distribution_properties(data.dropna().to_numpy())
