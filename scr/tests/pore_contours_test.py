import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
from os import path
from typing import Literal

from scr.config.paths import PATH_SUNSPOTS_PHASES

from scr.utils.types_alias import ObjectFilteringMode
from scr.geometry.contours.transform import shift_contours
from scr.geometry.crop.tight import crop_tight

from scr.io.fits.read import load_image

from scr.pipelines.io.load_phase_tracks import load_filtered_phase_tracks
from scr.postanalysis.selection.sunspots import apply_standard_sunspots_phases_filter

from scr.plotting.types import ContourGroup
from scr.plotting.composite.image_with_contours import plot_image_with_contours


MODE: ObjectFilteringMode = "pores"

contours_phases, df = load_filtered_phase_tracks(
    nosuffix_filename=path.join(PATH_SUNSPOTS_PHASES, "all_pores_phases_merged"),
    mode=MODE,
    drop_unknown=True,
)

df = df[df["umbra_component_count"] > 0]
df_corrupted = df[~np.isfinite(df["pore_component_count"])]  # should be empty, why it is not?
df = df[["observation_id", "sunspot_id", "phase", "frame", "image_path"]]

contours_phases = apply_standard_sunspots_phases_filter(contours_phases, df)

for iloc in np.random.choice(len(df), 20, replace=False):
    try:
        row = df.iloc[iloc]

        image = load_image(row["image_path"], quantity="Ic")

        image, (y_offset, x_offset) = crop_tight(
            image,
            contours_phases[row["observation_id"]][row["sunspot_id"]][row["phase"]]["outer"][row["frame"]],
            margin=20
        )

        contour_groups = []
        for color, region in [("blue", "outer"), ("red", "middle"), ("yellow", "inner")]:

             contour_groups.append(
                 ContourGroup(
                     contours=shift_contours(
                         contours_phases[row["observation_id"]][row["sunspot_id"]][row["phase"]][region][row["frame"]],
                         y_offset=y_offset, x_offset=x_offset
                     ),
                     style={
                         "color": color,
                         "linewidth": 1.5,
                         "linestyle": "-",
                     },
                     label=region,
                 )
             )

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_image_with_contours(
            ax=ax,
            image=image,
            contour_groups=contour_groups
        )
        plt.show(block=False)
        fig.savefig(f"pore_{iloc}.pdf")
    except:
        print(
            f"Contour does not exist: "
            f'{row["observation_id"]=}, {row["sunspot_id"]=}, {row["phase"]=}, {region=}, {row["frame"]=}'
        )
