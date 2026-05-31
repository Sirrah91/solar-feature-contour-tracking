from os import path
import numpy as np
from tqdm import tqdm

from scr.config.paths import PATH_SUNSPOTS_PHASES, PATH_VIDEOS
from scr.utils.filesystem import check_dir
from scr.utils.types_alias import Quantity, ObjectFilteringMode

from scr.pipelines.io.load_phase_tracks import load_filtered_phase_tracks
from scr.postanalysis.selection.sunspots import apply_standard_sunspots_phases_filter

from scr.plotting.scene.frame_spec import FrameSpec
from scr.plotting.animation.animate import animate_frames_from_generator
from scr.plotting.builders.frame_generators import iter_frame_data
from scr.plotting.scene.parsers import contour_groups_from_sunspot_phases

from scr.plotting.style.resolvers import sunspot_phase_style_resolver
from scr.plotting.style.colors import id_colors
from scr.plotting.annotate.contours import annotate_contour_groups_labels


def main():
    """
    Create an animation showing the temporal evolution of sunspot contours
    over intensity images for a single observation.
    """
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    QUANTITY: Quantity = "Ic"
    MODE: ObjectFilteringMode = "sunspots"
    PATH_SUNSPOTS_PHASES = "/nfsscratch/david/Contours/sunspots_removed_phases"

    obs_index = 0

    interval = 150  # ms between frames

    # ------------------------------------------------------------------
    # Load phase-tracked contours and metadata
    # ------------------------------------------------------------------
    contours_phases, df = load_filtered_phase_tracks(
        nosuffix_filename=path.join(PATH_SUNSPOTS_PHASES, "sunspots_phases"), filtering_options=MODE, drop_unknown=True)
    observation_ids = np.unique(df["observation_id"])

    for obs_index in tqdm(range(len(observation_ids))):
        observation_id = observation_ids[obs_index]
        out = path.join(PATH_VIDEOS, f"{QUANTITY}_{MODE}_{obs_index}.mp4")
        check_dir(out, is_file=True)

        df_obs = df[df["observation_id"] == observation_id]
        # del df

        contour_source = apply_standard_sunspots_phases_filter(
            contours_phases,
            df_obs,
        )
        # del contours_phases

        # ------------------------------------------------------------------
        # Define what constitutes a single animation frame
        # (unique image + frame index)
        # ------------------------------------------------------------------
        frame_df = (
            df_obs[["observation_id", "frame", "image_path"]]
            .drop_duplicates()
            .sort_values("image_path")
        )

        frame_specs = (
            FrameSpec(
                row.observation_id,
                row.frame,
                row.image_path,
            )
            for row in frame_df.itertuples(index=False)
        )

        # ------------------------------------------------------------------
        # Styling and annotations
        # ------------------------------------------------------------------
        sunspot_ids = np.unique(df_obs["sunspot_id"])
        colors = id_colors(sunspot_ids, cmap="gist_rainbow")

        style_resolver = sunspot_phase_style_resolver(
            colors=colors,
            linestyles={
                "forming": "--",
                "stable": "-",
                "decaying": ":",
            },
        )

        # ------------------------------------------------------------------
        # Build the frame generator
        # ------------------------------------------------------------------
        frame_iter = iter_frame_data(
            frame_specs=frame_specs,
            contour_source=contour_source,
            contour_parser=contour_groups_from_sunspot_phases,
            style_resolver=style_resolver,
            annotations=annotate_contour_groups_labels(fontsize=9),
            quantity=QUANTITY,
        )

        # ------------------------------------------------------------------
        # Animate and save
        # ------------------------------------------------------------------
        animate_frames_from_generator(
            frames=frame_iter,
            save_path=out,
            interval=interval,
        )


if __name__ == "__main__":
    main()
