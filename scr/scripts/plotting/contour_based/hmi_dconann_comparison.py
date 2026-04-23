import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from os import path
from astropy.wcs import WCS
from astropy.io import fits

from scr.config.figures import FIG_FORMAT, SAVEFIG_KWARGS
from scr.config.paths import PATH_FIGURES
from scr.config.quantities import _QUANTITIES
from scr.utils.filesystem import check_dir

from scr.geometry.contours.area import contour_area
from scr.geometry.contours.extraction import find_contours
from scr.geometry.crop.tight import crop_tight

from scr.plotting.types import ContourGroup
from scr.plotting.scene.frame_data import FrameData
from scr.plotting.scene.render import render_scene
from scr.plotting.style.fonts import font_style
from scr.plotting.style.colorbar import add_colorbar
from scr.plotting.style.ticks import axis_ticks_from_wcs


def main():
    """
    Compare deconvolved and original HMI continuum images for a single snapshot,
    showing matched intensity contours and helioprojective coordinates.
    """
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    crop_margin = 30

    figure_outdir = path.join(PATH_FIGURES, "paper_plots")
    check_dir(figure_outdir)

    # ------------------------------------------------------------------
    # Load FITS data
    # ------------------------------------------------------------------
    with fits.open(
        "/nfsscratch/david/Contours/fits_to_plot/AR-11084_20100702_S19E07/"
        "hmi.proc_720s_dconANN.20100702_040000_TAI.ibptr.fits"
    ) as hdu:
        dconANN = hdu[1].data[0]

    with fits.open(
        "/nfsscratch/david/Contours/fits_to_plot/AR-11084_20100702_S19E07/"
        "hmi.ic_720s.20100702_040000_TAI.1.continuum.fits"
    ) as hdu:
        hmi = hdu[1].data[0]
        header = hdu[0].header

    # ------------------------------------------------------------------
    # Determine region of interest from strongest deconvolved contour
    # ------------------------------------------------------------------
    dconANN_09 = sorted(
        find_contours(dconANN, level=0.9),
        key=contour_area,
        reverse=True,
    )[:1]

    # Crop both images using the same contour-defined window
    dconANN, (y_off, x_off) = crop_tight(
        dconANN,
        contours=dconANN_09,
        margin=crop_margin,
    )
    hmi, _ = crop_tight(
        hmi,
        contours=dconANN_09,
        margin=crop_margin,
    )

    # ------------------------------------------------------------------
    # Extract contours on cropped images
    # ------------------------------------------------------------------
    contour_groups = [
        ContourGroup(
            contours=sorted(
                find_contours(dconANN, level=0.5),
                key=contour_area,
                reverse=True,
            )[:1],
            style={"color": "yellow", "linewidth": 1.5, "linestyle": "--"},
            label="Deconvolved 0.5",
        ),
        ContourGroup(
            contours=sorted(
                find_contours(dconANN, level=0.9),
                key=contour_area,
                reverse=True,
            )[:1],
            style={"color": "green", "linewidth": 1.5, "linestyle": "--"},
            label="Deconvolved 0.9",
        ),
        ContourGroup(
            contours=sorted(
                find_contours(hmi, level=0.5),
                key=contour_area,
                reverse=True,
            )[:1],
            style={"color": "red", "linewidth": 1.5, "linestyle": "-"},
            label="SDO/HMI 0.5",
        ),
        ContourGroup(
            contours=sorted(
                find_contours(hmi, level=0.9),
                key=contour_area,
                reverse=True,
            )[:1],
            style={"color": "blue", "linewidth": 1.5, "linestyle": "-"},
            label="SDO/HMI 0.9",
        ),
    ]

    # ------------------------------------------------------------------
    # WCS adjustment and tick preparation
    # ------------------------------------------------------------------
    header["CRPIX1"] -= x_off
    header["CRPIX2"] -= y_off

    dx, dy = header["CDELT1"], header["CDELT2"]
    wcs = WCS(header)

    (x_ticks, x_labels), (y_ticks, y_labels) = axis_ticks_from_wcs(
        wcs=wcs,
        shape=hmi.shape,
        resolution=(dx, dy),
        step=10,
    )

    # ------------------------------------------------------------------
    # Global intensity scaling
    # ------------------------------------------------------------------
    vmin = np.nanmin((hmi, dconANN))
    vmax = np.nanmax((hmi, dconANN))

    frame_left = FrameData(
        image=hmi,
        contour_groups=contour_groups,
    )
    frame_right = FrameData(
        image=dconANN,
        contour_groups=contour_groups,
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    with font_style(fontsize=18):
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(
            1, 3,
            height_ratios=[1],
            width_ratios=[1, 1, 0.05],
            figure=fig
        )
        gs.update(wspace=0.05, hspace=0.05)

        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)
        cbar_ax = fig.add_subplot(gs[0, 2])

        render_scene(
            ax_left,
            image=frame_left.image,
            contour_groups=frame_left.contour_groups,
            vmin=vmin,
            vmax=vmax,
        )

        im = render_scene(
            ax_right,
            image=frame_right.image,
            contour_groups=frame_right.contour_groups,
            vmin=vmin,
            vmax=vmax,
            return_image=True,
        )

        # Avoid duplicated y-axis labels
        ax_right.tick_params(labelleft=False)

        add_colorbar(
            cbar_ax,
            im,
            label=_QUANTITIES["Ic"].latex,
            cbar_kwargs={"position": "right", "size": "100%", "pad": -0.2},
        )
        cbar_ax.axis("off")

        # Enforce identical physical aspect ratio
        ax_left.set_aspect(dx / dy)
        ax_right.set_aspect(dx / dy)

        for ax in (ax_left, ax_right):
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("Helioprojective Longitude (arcsec)")

        ax_left.set_yticks(y_ticks)
        ax_left.set_yticklabels(y_labels)
        ax_left.set_ylabel("Helioprojective Latitude (arcsec)")

        fig.savefig(
            path.join(figure_outdir, f"AR-11084.{FIG_FORMAT}"),
            format=FIG_FORMAT,
            **SAVEFIG_KWARGS,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
