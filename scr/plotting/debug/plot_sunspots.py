import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scr.utils.types_alias import Sunspots
from scr.geometry.contours.area import contour_area
from scr.plotting.types import ContourGroup
from scr.plotting.scene.plot_image import plot_image_with_contours


def plot_image_with_sunspots(
        image: np.ndarray,
        sunspots: Sunspots,
        *,
        frame: int,
        region: str = "all",
) -> tuple[Figure, Axes]:
    """
    Plot image with sunspot contours for a given frame.

    The label of each sunspot is placed on the largest region
    (largest contour area) of that sunspot.
    """
    # 1. Determine and sort unique regions for consistent coloring
    if region == "all":
        regions_found = {r for spot in sunspots.values() for r in spot.keys()}
    else:
        regions_found = {region}

    sorted_regions = sorted(regions_found)

    # 2. Assign colours

    colors = plt.get_cmap("brg_r")(np.linspace(0, 1, len(sorted_regions)))
    region_color = dict(zip(sorted_regions, colors))

    contour_groups: list[ContourGroup] = []

    # 3. Process each sunspot
    for sid, spot in sunspots.items():
        spot_contours_with_meta = []

        for r in sorted_regions:
            if r not in spot or frame not in spot[r]:
                continue

            # Store contours along with their region color for later styling
            for c in spot[r][frame]:
                spot_contours_with_meta.append({
                    "contour": c,
                    "color": region_color[r],
                    "area": contour_area(c)
                })

        if not spot_contours_with_meta:
            continue

        # . Find the single largest contour for the Sunspot ID label
        largest_item = max(spot_contours_with_meta, key=lambda x: x["area"])

        for item in spot_contours_with_meta:
            is_largest = (item is largest_item)

            contour_groups.append(
                ContourGroup(
                    contours=[item["contour"]],
                    style={
                        "color": item["color"],
                        "linewidth": 1.5,
                        "linestyle": "-",
                    },
                    # Only assign the label to the largest contour of the whole sunspot
                    label=str(sid) if is_largest else None,
                )
            )

    # 5. Render
    fig, ax = plot_image_with_contours(image, contour_groups)
    plt.tight_layout()
    plt.show(block=False)

    return fig, ax
