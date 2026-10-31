import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from os import path
import numpy as np

from src.config.paths import PATH_SUNSPOTS_PHASES, PATH_FIGURES
from src.config.figures import FIG_FORMAT, SAVEFIG_KWARGS
from src.io.fits.read import load_image
from src.pipelines.io.load_phase_tracks import load_filtered_phase_tracks
from src.postanalysis.selection.sunspots import apply_standard_sunspots_phases_filter

from src.plotting.types import ContourGroup
from src.plotting.scene.render import render_scene
from src.geometry.crop.centered import crop_centered_fixed
from src.geometry.contours.transform import shift_contours
from src.config.filtering import gimme_filtering_kwargs
from src.plotting.scene.frame_data import FrameData
from src.plotting.style.fonts import font_style

matplotlib.use("TkAgg")

PATH_SUNSPOTS_PHASES = "/nfsscratch/david/Contours/sunspots_removed_phases"

sunspots_phases, df = load_filtered_phase_tracks(
    nosuffix_filename=path.join(PATH_SUNSPOTS_PHASES, "sunspots_phases"),
    filtering_options=gimme_filtering_kwargs("pores")
                      | {"sunspot_mu_mean": {"mode": "frame-wise", "min_value": 0.8}}
                      | {"sunspot_area_corr": {"mode": "frame-wise", "min_value": 450.}},
    filter_phase_tracks=True,
    drop_unknown=True,
)

figure_outdir = path.join(PATH_FIGURES, "paper_plots")

unique_spot_global_indices = np.unique(df["spot_global_index"])
rng = np.random.default_rng(seed=42)

spot_global_indices = rng.choice(unique_spot_global_indices, 9, replace=False)

matplotlib.use("TkAgg")

CONTOUR_STYLES = {
    "Ic<0.5": {
        "color": "yellow",
        "linestyle": "-",
    },
    "Ic<0.65": {
        "color": "red",
        "linestyle": "-",
    },
    "Ic<0.9": {
        "color": "blue",
        "linestyle": "-",
    },
}

fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for ax, spot_global_index in zip(axes.flat, spot_global_indices):

    # All observations of this object
    df_obs = df[df["spot_global_index"] == spot_global_index]

    if df_obs.empty:
        ax.set_axis_off()
        continue

    contour_source = apply_standard_sunspots_phases_filter(
        sunspots_phases,
        df_obs,
    )

    # Select the largest appearance
    row = df_obs.loc[df_obs["sunspot_area_corr"].idxmax()]

    observation_id = row["observation_id"]
    sunspot_id = row["sunspot_id"]
    phase = row["phase"]
    frame = row["frame"]

    # Load image (however you currently do it)
    image = load_image(row["image_path"], quantity="Ic")

    sunspot = contour_source[observation_id][sunspot_id][phase]

    image, (y_offset, x_offset) = crop_centered_fixed(
        image,
        sunspot["Ic<0.9"][frame],
        target_shape=(100, 100),
    )

    contour_groups = []
    for part in sunspot:
        contour_groups.append(
            ContourGroup(
                contours=shift_contours(
                    sunspot[part][frame],
                    y_offset=y_offset,
                    x_offset=x_offset,
                ),
                style=CONTOUR_STYLES[part],
            )
        )


    render_scene(
        ax,
        image=image,
        contour_groups=contour_groups,
    )

    ax.axis("off")

plt.tight_layout()
plt.show(block=False)

fig.savefig(
    path.join(
        figure_outdir,
        f"pore_contours.{FIG_FORMAT}",
    ),
    format=FIG_FORMAT,
    **SAVEFIG_KWARGS,
)
plt.close(fig)
