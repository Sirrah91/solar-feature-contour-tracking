import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from os import path
import numpy as np

from scr.config.paths import PATH_SUNSPOTS_PHASES, PATH_FIGURES
from scr.config.figures import FIG_FORMAT, SAVEFIG_KWARGS
from scr.config.plotting import PHASE_COLORS
from scr.config.quantities import get_measurement_spec
from scr.utils.filesystem import check_dir
from scr.utils.types_alias import Quantity, SunspotPart, ObjectFilteringMode

from scr.geometry.crop.common import common_crop_shape_from_tracks
from scr.geometry.crop.centered import crop_centered_fixed
from scr.geometry.contours.extraction import find_contours
from scr.geometry.contours.transform import shift_contours
from scr.contours.selection import select_support_contours

from scr.io.fits.read import load_image
from scr.statistics.dataframe.timeseries import add_relative_time
from scr.pipelines.io.load_phase_tracks import load_filtered_phase_tracks
from scr.postanalysis.selection.sunspots import apply_standard_sunspots_phases_filter

from scr.plotting.types import ContourGroup
from scr.plotting.scene.render import render_scene
from scr.plotting.scene.frame_data import FrameData
from scr.plotting.style.fonts import font_style


def main():
    """
    Plot equidistant evolution snapshots of a single sunspot as a grid of images
    with phase-colored contours and support contours overlaid.
    """
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    QUANTITY: Quantity = "Ic"
    QUANTITY_SUPPORT: Quantity = "B"
    WHERE: SunspotPart = "sunspot"
    MODE: ObjectFilteringMode = "pores"
    PATH_SUNSPOTS_PHASES = "/nfsscratch/david/Contours/sunspots_removed_phases"

    spot_global_index = 72581

    nrows, ncols = 3, 5
    crop_margin = 30
    key_region: SunspotPart = "Ic<0.9"

    spec = get_measurement_spec(QUANTITY, WHERE)
    spec_support = get_measurement_spec(QUANTITY_SUPPORT, WHERE)

    figure_outdir = path.join(PATH_FIGURES, "paper_plots")
    check_dir(figure_outdir)

    # ------------------------------------------------------------------
    # Load phase tracks and select observation / sunspot
    # ------------------------------------------------------------------
    sunspots_phases, df = load_filtered_phase_tracks(
        nosuffix_filename=path.join(PATH_SUNSPOTS_PHASES, "sunspots_phases"), filtering_options=MODE, drop_unknown=True)

    df_obs = df[df["spot_global_index"] == spot_global_index]
    del df

    # Add time axis relative to first detection
    df_obs = add_relative_time(df_obs)

    # Select equidistant frames in time
    all_frames = df_obs["frame"]
    frame_indices = np.round(
        np.linspace(0, len(all_frames) - 1, nrows * ncols)
    ).astype(int)

    df_sel = df_obs.iloc[frame_indices].reset_index(drop=True)

    del df_obs

    # Restrict contour tracks to the selected frames only
    contour_source = apply_standard_sunspots_phases_filter(
        sunspots_phases,
        df_sel,
    )
    observation_id, sunspot_id = df_sel.iloc[0][["observation_id", "sunspot_id"]]
    sunspot_phases = contour_source[observation_id][sunspot_id]

    del sunspots_phases, contour_source

    # ------------------------------------------------------------------
    # Prepare FrameData objects (pure data, no plotting)
    # ------------------------------------------------------------------
    frames: list[FrameData] = []

    # Determine a common crop size across all selected frames
    target_shape = common_crop_shape_from_tracks(sunspot_phases, key_region=key_region, margin=crop_margin)

    for _, row in df_sel.iterrows():
        # Load primary and support images
        image = load_image(row["image_path"], quantity=QUANTITY)
        image_support = load_image(row["image_path"], quantity=QUANTITY_SUPPORT)

        # Primary contours for this frame
        contours = sunspot_phases[row["phase"]][key_region][row["frame"]]

        # Crop both images using the same fixed window
        image, (y_offset, x_offset) = crop_centered_fixed(
            image,
            contours,
            target_shape=target_shape,
        )
        image_support, _ = crop_centered_fixed(
            image_support,
            contours,
            target_shape=target_shape,
        )

        # Shift contours into the cropped image coordinate system
        contours = shift_contours(
            contours,
            y_offset=y_offset,
            x_offset=x_offset,
        )

        # Extract and select matching support contours
        support_contours_all = find_contours(
            image_support,
            level=spec_support.threshold,
        )
        support_contours = select_support_contours(
            contours,
            support_contours_all,
        )

        contour_groups = [
            ContourGroup(
                contours=contours,
                style={
                    "color": PHASE_COLORS[row["phase"]],
                    "linewidth": 1.5,
                    "linestyle": "-",
                },
                label=spec.quantity.latex,
            ),
            ContourGroup(
                contours=support_contours,
                style={
                    "color": "blue",
                    "linewidth": 1.0,
                    "linestyle": "-",
                },
                label=spec_support.quantity.latex,
            ),
        ]

        frames.append(
            FrameData(
                image=image,
                contour_groups=contour_groups,
            )
        )

    # ------------------------------------------------------------------
    # Global color scaling
    # ------------------------------------------------------------------
    vmin = np.nanmin([np.nanmin(frame.image) for frame in frames])
    vmax = np.nanmax([np.nanmax(frame.image) for frame in frames])

    # Pixel scale (arcsec per pixel, for aspect ratio only)
    dx, dy = 0.29714842896202537, 0.31997767629244117
    time_hours = df_sel["time_hours"]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    with font_style(fontsize=16):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 3, nrows * 3),
            squeeze=False,
        )
        axes = np.ravel(axes)

        for ax, frame, time_hour in zip(axes, frames, time_hours):
            render_scene(
                ax,
                image=frame.image,
                contour_groups=frame.contour_groups,
                vmin=vmin,
                vmax=vmax,
            )

            # Enforce physical aspect ratio
            ax.set_aspect(dx / dy)

            ax.set_title(rf"$t = {time_hour:.0f}$ h", pad=0)
            ax.axis("off")

        plt.tight_layout()

        fig.savefig(
            path.join(
                figure_outdir,
                f"evolution_snapshots_{QUANTITY}_{QUANTITY_SUPPORT}_{spot_global_index}.{FIG_FORMAT}",
            ),
            format=FIG_FORMAT,
            **SAVEFIG_KWARGS,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
