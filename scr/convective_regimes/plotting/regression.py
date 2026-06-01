from os import path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scr.config.figures import SAVEFIG_KWARGS
from scr.config.quantities import get_measurement_spec
from scr.convective_regimes.analysis.regression import analyse_regression_with_auc_loss
from scr.convective_regimes.core.thresholds import (
    BASELINE_WINDOWS,
    compute_thresholds,
)
from scr.convective_regimes.core.transitions import fit_transition_region
from scr.convective_regimes.io.filenames import regression_filename
from scr.convective_regimes.settings import DATA_DIR, FIG_FORMAT, FIGURE_DIR
from scr.plotting.style.fonts import font_style
from scr.utils.filesystem import check_dir
from scr.convective_regimes.utils.types_alias import FilterMode, SunspotPhase
from scr.io.parquet import load_parquet


class _PlotConfig(NamedTuple):
    """Plot-only geometry  analysis windows live in BASELINE_WINDOWS."""
    x_max: float
    baseline: str        # "ver" or "hor"  which component is primary
    loc: str
    show_transition: bool


_PLOT_CONFIGS: dict[tuple[FilterMode, str], _PlotConfig] = {
    ("sunspots", "PQ"): _PlotConfig(x_max=90.0, baseline="hor", loc="upper right", show_transition=True),
    ("sunspots", "UP"): _PlotConfig(x_max=50.0, baseline="ver", loc="lower right", show_transition=False),
    ("pores",    "PQ"): _PlotConfig(x_max=50.0, baseline="ver", loc="lower right", show_transition=False),
    ("pores",    "UP"): _PlotConfig(x_max=50.0, baseline="ver", loc="lower right", show_transition=False),
}

_TITLES = {
    "PQ": r"$I^\mathrm{{c}} = 0.9\,I^\mathrm{{c}}_\mathrm{{QS}}$",
    "UP": r"$I^\mathrm{{c}} = 0.5\,I^\mathrm{{c}}_\mathrm{{QS}}$",
}

# Maps (object_type, region, component) → attribute name on RegressionThresholds
_THRESHOLD_ATTR: dict[tuple[FilterMode, str, str], str] = {
    ("sunspots", "PQ", "hor"): "sunspot_pq_bhor",
    ("sunspots", "PQ", "ver"): "sunspot_pq_bver",
    ("sunspots", "UP", "ver"): "sunspot_up_bver",
    ("sunspots", "UP", "hor"): "sunspot_up_bhor",
    ("pores",    "PQ", "ver"): "pore_pq_bver",
    ("pores",    "PQ", "hor"): "pore_pq_bhor",
    ("pores",    "UP", "ver"): "pore_up_bver",
    ("pores",    "UP", "hor"): "pore_up_bhor",
}


def _load_and_augment(
        data_dir: str,
        object_type: FilterMode,
        region: str,
        phase: SunspotPhase,
        loss_tolerance: float,
) -> pd.DataFrame:
    filename = regression_filename(
        data_dir=data_dir,
        object_type=object_type,
        phase=phase,
        region=region,
    )

    df = load_parquet(
        filename,
        return_dataset=False
    )
    return analyse_regression_with_auc_loss(df, loss_tolerance=loss_tolerance)


def _compute_alphas(
        results: pd.DataFrame,
        alpha_floor: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    total_loss = results["loss_ver"] + results["loss_hor"]
    alphas_ver = np.where(total_loss > 0, results["loss_hor"] / total_loss, 0.5)
    alphas_hor = np.where(total_loss > 0, results["loss_ver"] / total_loss, 0.5)
    return (
        np.clip(alphas_ver, alpha_floor, 1.0),
        np.clip(alphas_hor, alpha_floor, 1.0),
    )


def plot_regression(
        phase: SunspotPhase = "all",
        *,
        data_dir: str = DATA_DIR,
        figure_outdir: str = FIGURE_DIR,
        fig_format: str = FIG_FORMAT,
        loss_tolerance: float = 0.01,
) -> None:
    """
    Plot sliding-window logistic regression results for all object types and boundaries.

    Produces one figure per (object_type, boundary) combination, saved to figure_outdir.
    """
    check_dir(figure_outdir)

    thresholds = compute_thresholds(phase, data_dir)

    spec_Binc = get_measurement_spec("Binc")
    spec_Bhor = get_measurement_spec("Bhor")
    spec_Bver = get_measurement_spec("Bver")

    for object_type in ("sunspots", "pores"):
        for region in ("PQ", "UP"):
            cfg = _PLOT_CONFIGS[(object_type, region)]
            window = BASELINE_WINDOWS[(object_type, region)]

            results = _load_and_augment(data_dir, object_type, region, phase, loss_tolerance)
            alphas_ver, alphas_hor = _compute_alphas(results)

            # Retrieve pre-computed estimate from thresholds
            attr = _THRESHOLD_ATTR[(object_type, region, cfg.baseline)]
            estimate = getattr(thresholds, attr)

            title = f"{object_type.title()} at {_TITLES[region]}"
            print(f"{title} ({phase}): B{cfg.baseline} = {estimate.mean:.0f} ± {estimate.std:.0f} G")

            with font_style(fontsize=16):
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.set_title(title)
                ax.set_xlabel(spec_Binc.label())
                ax.set_ylabel(
                    spec_Bver.label(superscript="crit", depends_on=[spec_Binc.quantity]),
                    color="red",
                )
                ax.tick_params(axis="y", labelcolor="red")

                ax2 = ax.twinx()
                ax2.set_ylabel(
                    spec_Bhor.label(superscript="crit", depends_on=[spec_Binc.quantity]),
                    color="blue",
                )
                ax2.tick_params(axis="y", labelcolor="blue")

                ax.set_xlim(left=0.0, right=cfg.x_max)

                for i in range(len(results) - 1):
                    seg_alpha_ver = 0.5 * (alphas_ver[i] + alphas_ver[i + 1])
                    seg_alpha_hor = 0.5 * (alphas_hor[i] + alphas_hor[i + 1])

                    ax.plot(
                        results["gamma_center"].iloc[i:i + 2],
                        results["Bver_from_combined"].iloc[i:i + 2],
                        color="red", alpha=seg_alpha_ver, linewidth=2.5,
                    )
                    ax2.plot(
                        results["gamma_center"].iloc[i:i + 2],
                        results["Bhor_from_combined"].iloc[i:i + 2],
                        color="blue", alpha=seg_alpha_hor, linewidth=2.5,
                    )

                gamma_mask = (
                        (results["gamma_center"] <= cfg.x_max)
                        & (results["gamma_center"] >= 0.0)
                )
                top = np.max((
                    np.nanmax(results["Bhor_from_combined"][gamma_mask]),
                    np.nanmax(results["Bver_from_combined"][gamma_mask]),
                )) * 1.1
                ax.set_ylim(bottom=0.0, top=top)
                ax2.set_ylim(bottom=0.0, top=top)

                lns = []
                lns += ax.plot([], [], color="red", linewidth=2.5,
                               label=f"{spec_Bver.quantity.latex} component")
                lns += ax2.plot([], [], color="blue", linewidth=2.5,
                                label=f"{spec_Bhor.quantity.latex} component")

                if estimate.is_valid:
                    xmin_frac = window.gamma_min / cfg.x_max
                    xmax_frac = window.gamma_max / cfg.x_max
                    spec = spec_Bver if cfg.baseline == "ver" else spec_Bhor
                    comp_ax = ax if cfg.baseline == "ver" else ax2
                    line_color = "darkred" if cfg.baseline == "ver" else "darkblue"

                    line_handle = comp_ax.axhline(
                        y=estimate.mean,
                        xmin=xmin_frac,
                        xmax=xmax_frac,
                        color=line_color,
                        linestyle="--",
                        linewidth=2.0,
                        label=(
                            rf"Baseline $\langle {spec.quantity.latex[1:-1]}"
                            rf"\rangle = {estimate.mean:.0f}\,\mathrm{{{spec.quantity.unit}}}$"
                        ),
                    )
                    lns.append(line_handle)

                if cfg.show_transition:
                    transition = fit_transition_region(
                        gamma_centers=results["gamma_center"].to_numpy(),
                        loss_ver=results["loss_ver"].to_numpy(),
                        loss_hor=results["loss_hor"].to_numpy(),
                    )

                    print(f"{title} ({phase}): Binc = {transition.start_deg:.1f} - {transition.end_deg:.1f} deg")

                    ax.axvspan(
                        transition.start_deg, transition.end_deg,
                        color="purple", alpha=0.07, label="Transition zone",
                    )
                    ax.axvline(x=transition.start_deg, color="purple",
                               linestyle=":", linewidth=1.5, alpha=0.6)
                    ax.axvline(x=transition.end_deg, color="purple",
                               linestyle=":", linewidth=1.5, alpha=0.6)

                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs, loc=cfg.loc)

                fig.savefig(
                    path.join(figure_outdir, f"{object_type}_{phase}_{region}__regression.{fig_format}"),
                    format=fig_format,
                    **SAVEFIG_KWARGS,
                )
                plt.close(fig)
