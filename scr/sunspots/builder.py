import numpy as np
from tqdm import tqdm

from scr.utils.types_alias import Sunspots, TracksLabeled, SunspotPart, AssociationMode
from scr.utils.dict import separate_key_value
from scr.utils.collections import nested_defaultdict

from scr.geometry.contours.extraction import extract_frame_contours
from scr.geometry.contours.relations import contour_belongs_to_outer
from scr.geometry.contours.shapes import contour_to_shape, contours_to_shape, prepare_shape


def build_sunspots_from_outer_tracks(
        images: np.ndarray,
        outer_tracks: TracksLabeled,
        levels: dict[SunspotPart, float],
        *,
        min_area: dict[SunspotPart, float] | None = None,
        containment_mode: AssociationMode = "covers",
        min_containment: float = 0.8,
) -> Sunspots:
    """
    Build sunspot dictionary by extracting nested contours inside tracked
    outer boundaries.

    Each sunspot at a given frame is defined by the union of all its
    outer contours (can be multiple disjoint components).

        Parameters
    ----------
    images : ndarray (T, H, W)
    outer_tracks : Tracks
        Output of track_contours at the outer level.
    levels : dict
        Mapping component -> contour level, e.g.
        {"middle": 0.65, "inner": 0.5}
    min_area : dict, optional
        Per-component minimum contour area.
    containment_mode : str
        Passed to contour_belongs_to_outer ("covers", "intersects", ...)
    min_containment : float
        Minimum fraction of the smaller region that must be inside the larger one
        (in case of `containment_mode=="robust"`).

    Returns
    -------
    Sunspots
    """

    min_area = min_area or {}

    sunspots: Sunspots = nested_defaultdict(factory=list, depth=3)
    outer_name, outer_tracks = separate_key_value(outer_tracks)

    for sid, track in tqdm(outer_tracks.items(), desc="Processing outer tracks → sunspots", unit="track"):

        for t, outer_contours in track.items():

            image = images[t]

            # --- 1. Convert all outer contours to union shape
            union_shape = prepare_shape(contours_to_shape(outer_contours))

            # --- 2. Store outer contours (as they are)
            sunspots[sid][outer_name][t].extend(outer_contours)

            # --- 3. Extract inner levels only once per frame
            for name, level in levels.items():

                contours = extract_frame_contours(
                    image=image,
                    level=level,
                    min_area=min_area.get(name, 0.0),
                )

                matched = []

                for c in contours:
                    if contour_belongs_to_outer(
                        inner=contour_to_shape(c),
                        outer=union_shape,
                        mode=containment_mode,
                        min_fraction=min_containment,
                    ):
                        matched.append(c)

                if matched:
                    sunspots[sid][name][t].extend(matched)

    return sunspots
