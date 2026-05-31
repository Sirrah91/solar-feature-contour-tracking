from os import path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm

from scr.config.figures import CBAR_KWARGS, SAVEFIG_KWARGS
from scr.convective_regimes.core.regions import SUNSPOT_PQ_B_THRESHOLD
from scr.convective_regimes.core.thresholds import RegressionThresholds, compute_thresholds
from scr.convective_regimes.io.loaders import load_probability_map
from scr.convective_regimes.io.filenames import probability_filename
from scr.convective_regimes.settings import DATA_DIR, FIG_FORMAT, FIGURE_DIR
from scr.convective_regimes.utils.types_alias import SunspotPhase
from scr.plotting.generic.image import plot_image
from scr.plotting.style.lines import LineSpec
from scr.plotting.style.colorbar import add_colorbar
from scr.plotting.style.fonts import font_style
from scr.utils.filesystem import check_dir


def _build_contour_levels(t: RegressionThresholds) -> dict[str, dict]:
    """
    Build per-object contour specs from computed thresholds.

    Each entry is a list of LineSpec so level, color, linestyle, and linewidth
    are always paired. Missing (invalid) estimates are silently dropped.
    """
    return {
        "sunspots": {
            "Bver": [LineSpec(1000.0, "red", "--")] + [
                s for s, e in (
                    (LineSpec(t.sunspot_up_bver.mean, "red", "-"),  t.sunspot_up_bver),
                )
                if e.is_valid
            ],
            "B": [
                LineSpec(SUNSPOT_PQ_B_THRESHOLD, "magenta", "-"),
            ],
            "Bhor": [
                s for s, e in (
                    (LineSpec(t.sunspot_pq_bhor.mean, "blue", "-"), t.sunspot_pq_bhor),
                )
                if e.is_valid
            ],
        },
        "pores": {
            "Bver": [
                s for s, e in (
                    (LineSpec(t.pore_pq_bver.mean, "red", "--"), t.pore_pq_bver),
                    (LineSpec(t.pore_up_bver.mean, "red", "-"),  t.pore_up_bver),
                )
                if e.is_valid
            ],
            "B": [
                LineSpec(SUNSPOT_PQ_B_THRESHOLD, "magenta", "-"),
            ],
            "Bhor": [],
        },
    }


def plot_probability_counts(
        phase: SunspotPhase = "all",
        *,
        data_dir: str = DATA_DIR,
        figure_outdir: str = FIGURE_DIR,
        fig_format: str = FIG_FORMAT,
        region: str = "penumbra",
) -> None:
    """
    Plot 2D penumbra probability and count maps.

    Iso-contour levels are computed from regression data via compute_thresholds,
    consistent with plot_regression and plot_2d_histograms.
    """
    check_dir(figure_outdir)

    thresholds = compute_thresholds(phase, data_dir)
    contour_levels = _build_contour_levels(thresholds)

    for use_counts in (True, False):
        for object_type in ("sunspots", "pores"):
            filename = probability_filename(
                data_dir=data_dir,
                object_type=object_type,
                phase=phase,
                region=region,
                quantity="Bhor",
            )
            pmap = load_probability_map(filename)
            cfg = contour_levels[object_type]

            # Derived iso-curves in (Bhor, gamma) space
            Bhor = pmap.x_centers[:, None]
            Gamma = pmap.y_centers[None, :]
            Bver = Bhor / np.tan(np.deg2rad(Gamma))
            B = Bhor / np.sin(np.deg2rad(Gamma))

            val = pmap.counts if use_counts else pmap.probability
            if use_counts:
                val = val.copy()
                val[val == 0] = np.nan

            invalid = ~np.isfinite(pmap.probability)
            Bver[invalid] = np.nan
            B[invalid] = np.nan

            with font_style(fontsize=16):
                fig, ax = plt.subplots(figsize=(8, 6))

                im = plot_image(
                    ax, val.T,
                    cmap="viridis",
                    vmin=None if use_counts else 0.0,
                    vmax=None if use_counts else 1.0,
                    image_kwargs={
                        "extent": (
                            pmap.x_bins[0], pmap.x_bins[-1],
                            pmap.y_bins[0], pmap.y_bins[-1],
                        ),
                        "aspect": "auto",
                        "norm": PowerNorm(gamma=0.5, vmin=0.0) if use_counts else None,
                    },
                )

                X, Y = np.meshgrid(pmap.x_centers, pmap.y_centers)

                legend_handles = []
                legend_labels = []

                for spec in cfg["B"]:
                    cs = ax.contour(X, Y, B.T, levels=[spec.level],
                                    colors=spec.color, linestyles=spec.linestyle,
                                    linewidths=spec.linewidth)
                    h, _ = cs.legend_elements()
                    legend_handles += h
                    legend_labels.append(
                        rf"$B = {spec.level:.0f}\,\mathrm{{G}}$"
                    )

                for spec in cfg["Bver"]:
                    cs = ax.contour(X, Y, Bver.T, levels=[spec.level],
                                    colors=spec.color, linestyles=spec.linestyle,
                                    linewidths=spec.linewidth)
                    h, _ = cs.legend_elements()
                    legend_handles += h
                    legend_labels.append(
                        rf"$B_{{\mathrm{{ver}}}} = {spec.level:.0f}\,\mathrm{{G}}$"
                    )

                for spec in cfg["Bhor"]:
                    h = ax.axvline(x=spec.level, color=spec.color,
                                   linestyle=spec.linestyle, linewidth=spec.linewidth)
                    legend_handles.append(h)
                    legend_labels.append(
                        rf"$B_{{\mathrm{{hor}}}} = {spec.level:.0f}\,\mathrm{{G}}$"
                    )

                ax.set_xlabel(rf"$B_{{\mathrm{{hor}}}} \, \left( \mathrm{{G}} \right)$")
                ax.set_ylabel(r"$\gamma \, \left( \mathrm{deg} \right)$")
                ax.set_xlim(left=0, right=4500)
                ax.set_ylim(bottom=0, top=90)
                ax.set_xticks(np.arange(0, 5000, 1000).astype(int))
                ax.set_title(f"{object_type.title()}")

                reg_label = (
                    r"0.5\,I^\mathrm{c}_\mathrm{QS} < I^\mathrm{c}"
                    r" \leq 0.9\,I^\mathrm{c}_\mathrm{QS}"
                )
                cbar = add_colorbar(
                    ax, im,
                    cbar_kwargs=CBAR_KWARGS,
                    label=(
                        rf"${'N' if use_counts else 'P'}"
                        rf"\left( {reg_label} \right)$"
                    ),
                )
                if use_counts:
                    cbar.formatter.set_powerlimits((0, 0))

                ax.legend(legend_handles, legend_labels, loc="lower right")

                suffix = "counts" if use_counts else "probability"
                fig.savefig(
                    path.join(
                        figure_outdir,
                        f"{object_type}_{phase}_{region}_Bhor_{suffix}.{fig_format}",
                    ),
                    format=fig_format,
                    **SAVEFIG_KWARGS,
                )
                plt.close(fig)
