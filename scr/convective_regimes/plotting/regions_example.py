from __future__ import annotations

from os import path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from scr.config.figures import SAVEFIG_KWARGS
from scr.config.filtering import gimme_filtering_kwargs
from scr.contours.selection import select_support_contours
from scr.convective_regimes.settings import (
    FIG_FORMAT, FIGURE_DIR, SUNSPOTS_PHASES_DIR_TEMPLATE,
)
from scr.geometry.contours.extraction import find_contours
from scr.geometry.raster.api import rasterize
from scr.io.fits.read import load_image
from scr.physics.magnetic import compute_Binc, compute_unsigned_inclination
from scr.pipelines.io.load_phase_tracks import load_filtered_phase_tracks
from scr.plotting.generic.contours import plot_contours
from scr.plotting.generic.image import plot_image
from scr.plotting.style.fonts import font_style
from scr.utils.filesystem import check_dir
from scr.convective_regimes.core.regions import (
    LIGHT_BRIDGE_REGION,
    TRANSITION_REGION,
    PORE_GAP_REGION,
)
from scr.convective_regimes.utils.types_alias import FilterMode


def plot_regions_example(
    *,
    sunspot_type: str = "removed",
    sunspots_phases_dir: str | None = None,
    filter_mode: FilterMode = "sunspots",
    level: float = 605.0,
    image_filter: str = ".*AR-11113_.*20101022_1300.*",
    figure_outdir: str = FIGURE_DIR,
    fig_format: str = FIG_FORMAT,
    xlim: tuple[int, int] = (328, 508),
    ylim: tuple[int, int] = (214, 394),
) -> None:
    """
    Plot a representative image with coloured region mask overlays.

    Highlights three regimes: light bridges (yellow), transition region (red),
    and pore-like vertical-field region (blue).
    """
    check_dir(figure_outdir)

    if sunspots_phases_dir is None:
        sunspots_phases_dir = SUNSPOTS_PHASES_DIR_TEMPLATE.format(
            sunspot_type=sunspot_type
        )

    sunspots_phases, df = load_filtered_phase_tracks(
        nosuffix_filename=path.join(sunspots_phases_dir, "sunspots_phases"),
        filtering_options={
            "image_path": {
                "mode": "frame-wise",
                "func": lambda x: x.str.contains(image_filter, na=False),
            }
        } | gimme_filtering_kwargs(filter_mode),
        filter_phase_tracks=True,
        drop_unknown=True,
    )

    df = df[["image_path", "observation_id", "frame", "sunspot_id", "phase"]]

    for _, (image_path, group) in enumerate(df.groupby(["image_path"], observed=True)):
        image_path = image_path[0]

        images = {
            "Ic": load_image(image_path, quantity="Ic"),
            "B": load_image(image_path, quantity="B"),
            "Bhor": load_image(image_path, quantity="Bhor"),
            "Br": np.abs(load_image(image_path, quantity="Br")),
        }
        images["Binc"] = np.rad2deg(
            compute_unsigned_inclination(
                signed_inclination=compute_Binc(
                    Br=images["Br"], Bhor=images["Bhor"]
                )
            ).astype(np.float32)
        )

        image_shape = images["B"].shape
        b_contours_all = find_contours(images["B"], level=level)

        penumbra_contours = []
        for iloc in range(len(group)):
            row = group.iloc[iloc]
            penumbra_contours.extend(
                sunspots_phases[row["observation_id"]]
                [row["sunspot_id"]][row["phase"]]["Ic<0.9"][row["frame"]]
            )

        b_contours = select_support_contours(penumbra_contours, b_contours_all)
        b_mask = rasterize(
            b_contours, image_shape,
            mode="surface", engine="skimage",
            use_orientation=False, dtype=bool,
        )

        with font_style(fontsize=16):
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_image(ax, images["Ic"])

            for region, color, lw in zip(
                [LIGHT_BRIDGE_REGION, TRANSITION_REGION, PORE_GAP_REGION],
                ["yellow", "red", "blue"],
                [3, 2, 1],
            ):
                mask = region.interior(
                    Bhor=images["Bhor"],
                    Binc=images["Binc"],
                    Br=images["Br"],
                    extra_mask=b_mask,
                )

                if np.any(mask):
                    plot_contours(
                        ax,
                        find_contours(mask.astype(float), level=0.5),
                        contour_kwargs={"color": color, "linewidth": lw},
                    )

            ax.axis("off")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            fig.savefig(
                path.join(figure_outdir, f"regions_example.{fig_format}"),
                format=fig_format,
                **SAVEFIG_KWARGS,
            )
            plt.close(fig)
