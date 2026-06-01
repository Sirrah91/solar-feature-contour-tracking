from os import path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from scr.config.figures import SAVEFIG_KWARGS
from scr.config.quantities import get_measurement_spec
from scr.convective_regimes.analysis.filling import analyse_penumbral_filling_vs_flux
from scr.convective_regimes.io.loaders import extract_q, load_all
from scr.convective_regimes.settings import DATA_DIR, FIG_FORMAT, FIGURE_DIR
from scr.geometry.solar.units import pixelarea_to_Mm2
from scr.plotting.style.fonts import font_style
from scr.utils.filesystem import check_dir
from scr.convective_regimes.utils.types_alias import SunspotPhase, FilterMode
from scr.convective_regimes.core.regions import TRANSITION_REGION, PENUMBRA


def plot_flux_in_target_region(
        phase: SunspotPhase = "all",
        *,
        data_dir: str = DATA_DIR,
        figure_outdir: str = FIGURE_DIR,
        fig_format: str = FIG_FORMAT,
        n_flux_bins: int = 100,
) -> None:
    """
    Plot penumbral regime filling factor as a function of total magnetic flux.

    Two curves: pixels in the transition regime (with penumbra) vs outside it.
    """
    check_dir(figure_outdir)

    all_data: list[FilterMode] = ["sunspots", "pores"]

    kwargs = dict(
        gamma=extract_q(load_all(all_data, [phase], data_dir=data_dir), q="Binc", per_object=True),
        Ic=extract_q(load_all(all_data, [phase], data_dir=data_dir), q="Ic", per_object=True),
        B=extract_q(load_all(all_data, [phase], data_dir=data_dir), q="Bhor", per_object=True),
        Phi=extract_q(load_all(all_data, [phase], data_dir=data_dir), q="Phi", per_object=True),
        n_flux_bins=n_flux_bins,
        ic_boundary=np.inf,
    )

    with_pen_res = analyse_penumbral_filling_vs_flux(
        **kwargs,
        region_function=lambda Bhor, Binc, Ic: TRANSITION_REGION.interior(
            Bhor=Bhor, Binc=Binc, extra_mask=PENUMBRA.interior(Ic=Ic)
        ),
    )

    no_pen_res = analyse_penumbral_filling_vs_flux(
        **kwargs,
        region_function=lambda Bhor, Binc, Ic: TRANSITION_REGION.interior(
            Bhor=Bhor, Binc=Binc, extra_mask=PENUMBRA.exterior(Ic=Ic)
        ),
    )

    pore_threshold = (
            np.nanmax(
                extract_q(load_all(["pores"], [phase], data_dir=data_dir), q="Phi")
            )
            * pixelarea_to_Mm2(1.0) * 10 ** 16
    )

    spec_flux = get_measurement_spec("Phi")
    spec_B = get_measurement_spec("B")

    with font_style(fontsize=16):
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax2 = ax1.twinx()

        ax1.plot(with_pen_res["flux_centers"], with_pen_res["median_filling"],
                 color="red", linestyle="-")
        ax1.fill_between(
            with_pen_res["flux_centers"],
            with_pen_res["p16"], with_pen_res["p84"],
            color="red", alpha=0.2,
        )

        ax2.plot(no_pen_res["flux_centers"], no_pen_res["median_filling"],
                 color="blue", linestyle="-")
        ax2.fill_between(
            no_pen_res["flux_centers"],
            no_pen_res["p16"], no_pen_res["p84"],
            color="blue", alpha=0.2,
        )

        ax1.axvline(x=pore_threshold, color="black", linestyle="--")

        ax1.set_xlabel(
            rf"Magnetic flux enclosed by the "
            rf"${spec_B.quantity.latex[1:-1]} = {605}\,\mathrm{{{spec_B.quantity.unit}}}$"
            rf" contour\,({spec_flux.quantity.unit})"
        )
        ax1.set_ylabel(
            r"Median filling factor of target region"
            "\n"
            r"for $0.5\,I^\mathrm{c}_\mathrm{QS} < I^\mathrm{c}"
            r"\leq 0.9\,I^\mathrm{c}_\mathrm{QS}$",
            color="red",
        )
        ax1.tick_params(axis="y", labelcolor="red")
        ax1.set_ylim(bottom=0)
        ax1.set_xlim(left=0)
        ax1.grid(False)

        ax2.set_ylabel(
            r"Median filling factor of target region"
            "\n"
            r"for $I^\mathrm{c} \leq 0.5\,I^\mathrm{c}_\mathrm{QS}$"
            r" or $I^\mathrm{c} > 0.9\,I^\mathrm{c}_\mathrm{QS}$",
            color="blue",
        )
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2.set_ylim(bottom=0)
        ax2.grid(False)

        fig.savefig(
            path.join(figure_outdir, f"flux_in_target_region_{phase}.{fig_format}"),
            format=fig_format,
            **SAVEFIG_KWARGS,
        )
        plt.close(fig)
