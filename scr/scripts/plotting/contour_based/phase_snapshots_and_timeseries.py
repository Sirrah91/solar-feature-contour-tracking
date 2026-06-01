import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
from scr.geometry.contours.transform import shift_contours

from scr.io.fits.read import load_image
from scr.statistics.dataframe.timeseries import add_relative_time
from scr.pipelines.io.load_phase_tracks import load_filtered_phase_tracks
from scr.postanalysis.selection.sunspots import apply_standard_sunspots_phases_filter
from scr.postanalysis.selection.representative_frames import select_representative_frames

from scr.plotting.types import ContourGroup
from scr.plotting.timeseries.colored import add_colored_timeseries
from scr.plotting.scene.render import render_scene
from scr.plotting.scene.frame_data import FrameData
from scr.plotting.style.fonts import font_style


def main() -> None:
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    QUANTITY_SNAPSHOTS: Quantity = "Ic"
    QUANTITY_LINEPLOTS: Quantity = "B"
    WHERE: SunspotPart = "sunspot"
    MODE: ObjectFilteringMode = "sunspots"
    PATH_SUNSPOTS_PHASES = "/nfsscratch/david/Contours/sunspots_removed_phases"

    spot_global_index = 72581

    crop_margin = 20
    key_region: SunspotPart = "Ic<0.9"

    spec_snapshots = get_measurement_spec(QUANTITY_LINEPLOTS, WHERE)
    spec_lineplots = get_measurement_spec(QUANTITY_LINEPLOTS, WHERE)

    figure_outdir = path.join(PATH_FIGURES, "paper_plots")
    check_dir(figure_outdir)

    # ----------------------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------------------
    contours_phases, df = load_filtered_phase_tracks(
        nosuffix_filename=path.join(PATH_SUNSPOTS_PHASES, "sunspots_phases"), filtering_options=MODE, drop_unknown=True)

    df_obs = df[df["spot_global_index"] == spot_global_index]
    del df

    # Add time axis relative to first detection
    df_obs = add_relative_time(df_obs)

    # Select representative frames in phases
    frame_indices = select_representative_frames(df_obs)
    df_sel = (
        df_obs[df_obs["frame"].isin(frame_indices)]
        .sort_values("frame")  # ensures frame order
        .reset_index(drop=True)
    )

    # Restrict contour tracks to the selected frames only
    contour_source = apply_standard_sunspots_phases_filter(
        contours_phases,
        df_sel,
    )
    observation_id, sunspot_id = df_sel.iloc[0][["observation_id", "sunspot_id"]]
    sunspot_phases = contour_source[observation_id][sunspot_id]

    del contours_phases, contour_source

    # ------------------------------------------------------------------
    # Prepare FrameData objects (pure data, no plotting)
    # ------------------------------------------------------------------
    frames: list[FrameData] = []

    # Determine a common crop size across all selected frames
    target_shape = common_crop_shape_from_tracks(sunspot_phases, key_region=key_region, margin=crop_margin)

    for _, row in df_sel.iterrows():
        # Load the image
        image = load_image(row["image_path"], quantity=QUANTITY_SNAPSHOTS)

        # Contours for this frame
        contours = sunspot_phases[row["phase"]][key_region][row["frame"]]

        # Crop the image using the fixed window
        image, (y_offset, x_offset) = crop_centered_fixed(
            image,
            contours,
            target_shape=target_shape,
        )

        # Shift contours into the cropped image coordinate system
        contours = shift_contours(
            contours,
            y_offset=y_offset,
            x_offset=x_offset,
        )

        contour_groups = [
            ContourGroup(
                contours=contours,
                style={
                    "color": PHASE_COLORS[row["phase"]],
                    "linewidth": 1.5,
                    "linestyle": "-",
                },
                label=spec_snapshots.quantity.latex,
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

    phases_titles = df_sel["phase"].str.title()
    time_hours = df_obs["time_hours"]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    with font_style(fontsize=12):
        nrows, ncols = 2, 3  # in principle can be more

        fig = plt.figure(figsize=(4 * ncols, 4 * nrows))

        gs = GridSpec(
            nrows, ncols,
            height_ratios=[1] * nrows,
            figure=fig
        )
        gs.update(wspace=0.05, hspace=0.05)

        top_axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
        line_axes = [fig.add_subplot(gs[i, :]) for i in range(1, nrows)]

        for ax in top_axes:
            ax.set_aspect("equal", adjustable="box")
        for ax in top_axes + line_axes:
            ax.margins(0)
            ax.set_anchor("C")

        # ---------------------------------------------------------------------
        # Top row: representative snapshots
        # ---------------------------------------------------------------------
        for ax, frame, title in zip(top_axes, frames, phases_titles):
            render_scene(
                ax,
                image=frame.image,
                contour_groups=frame.contour_groups,
                vmin=vmin,
                vmax=vmax,
            )

            # Enforce physical aspect ratio
            ax.set_aspect(dx / dy)

            ax.set_title(title)

            # No ax.axis("off") to keep the boundary box
            ax.set_xticks([])
            ax.set_yticks([])

        # ---------------------------------------------------------------------
        # Bottom row: time series (mean + std on twin axis)
        # ---------------------------------------------------------------------
        for ax_ts in line_axes:
            add_colored_timeseries(
                ax_ts,
                x=time_hours,
                y=df_obs[spec_lineplots.mean_col],
                phases=df_obs["phase"],
                phase_colors=PHASE_COLORS,
                linestyle="-",
                linewidth=2.0,
            )

            if spec_lineplots.threshold is not None:
                ax_ts.axhline(
                    y=spec_lineplots.threshold,
                    xmin=np.nanmin(time_hours),
                    xmax=np.nanmax(time_hours),
                    color="k",
                    linestyle="--"
                )

            # Mark snapshot times
            for t in df_sel["time_hours"]:
                ax_ts.axvline(
                    x=t,
                    color="k",
                    linestyle="--"
                )

            ax_ts.set_xlabel(r"Time since first detection\,(h)")
            ax_ts.set_ylabel(spec_lineplots.label_mean())

            margin = np.nanmax(df_obs[spec_lineplots.mean_col]) * 0.05
            bottom = max(0, np.nanmin(df_obs[spec_lineplots.mean_col]) - margin)
            top = np.nanmax(df_obs[spec_lineplots.mean_col]) + margin
            ax_ts.set_ylim(bottom, top)

            # ax_ts.grid(alpha=0.3)

            ax_std = ax_ts.twinx()
            add_colored_timeseries(
                ax_std,
                x=time_hours,
                y=df_obs[spec_lineplots.std_col],
                phases=df_obs["phase"],
                phase_colors=PHASE_COLORS,
                linestyle=":",
                linewidth=2.0,
            )

            ax_std.set_ylabel(spec_lineplots.label_std)

            margin = np.nanmax(df_obs[spec_lineplots.std_col]) * 0.05
            bottom = max(0, np.nanmin(df_obs[spec_lineplots.std_col]) - margin)
            top = np.nanmax(df_obs[spec_lineplots.std_col]) + margin
            ax_std.set_ylim(bottom, top)

            # Phase legend (proxy artists)
            for phase, color in PHASE_COLORS.items():
                if phase in df_obs["phase"].values:
                    ax_ts.plot([], [], color=color, label=phase.title(), lw=3)
            ax_ts.legend(loc="upper center", ncol=3)  # 3 or fewer columns

        fig.savefig(
            path.join(
                figure_outdir,
                f"timeseries_with_snapshots_{QUANTITY_SNAPSHOTS}_{QUANTITY_LINEPLOTS}_{spot_global_index}.{FIG_FORMAT}",
            ),
            format=FIG_FORMAT,
            **SAVEFIG_KWARGS,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
