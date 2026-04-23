import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from scr.plotting.types import ContourGroup
from scr.plotting.scene.render import render_scene
from scr.plotting.scene.frame_data import FrameData
from scr.plotting.style.fonts import font_style
from scr.plotting.annotate.contours import annotate_contour_groups_labels


def plot_image_with_contours(
        image: np.ndarray,
        contour_groups: list[ContourGroup]
) -> tuple[Figure, Axes]:
    """
    Plot equidistant evolution snapshots of a single sunspot as a grid of images
    with phase-colored contours and support contours overlaid.
    """

    frame = FrameData(
        image=image,
        contour_groups=contour_groups,
        annotations=annotate_contour_groups_labels(fontsize=9),
    )

    vmin = np.nanmin(frame.image)
    vmax = np.nanmax(frame.image)

    with font_style(fontsize=16):
        fig, axis = plt.subplots(
            1,
            1,
            figsize=(8, 6),
        )
        render_scene(
            axis,
            image=frame.image,
            contour_groups=frame.contour_groups,
            vmin=vmin,
            vmax=vmax,
            annotations=frame.annotations
        )

    return fig, axis
