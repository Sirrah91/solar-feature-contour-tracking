import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scr.utils.types_alias import Tracks
from scr.plotting.types import ContourGroup
from scr.plotting.scene.plot_image import plot_image_with_contours


def plot_image_with_tracks(
        image: np.ndarray,
        tracks: Tracks,
        *,
        frame: int,
) -> tuple[Figure, Axes]:
    """
    Plot image with track contours for a given frame.

    The label of each track is placed in the centroid of that track.
    """

    contour_groups = []

    for tid, frames in tracks.items():

        if frame not in frames:
            continue

        contour_groups.append(
            ContourGroup(
                contours=frames[frame],
                style={
                    "color": "red",
                    "linewidth": 1.5,
                    "linestyle": "-",
                },
                label=str(tid),
            )
        )

    fig, ax = plot_image_with_contours(image, contour_groups)

    plt.tight_layout()
    plt.show(block=False)

    return fig, ax
