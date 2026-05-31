import numpy as np
from tqdm import tqdm

from scr.utils.types_alias import Sunspots, TracksLabeled, SunspotPart, AssociationMode
from scr.utils.dict import separate_key_value
from scr.utils.collections import nested_defaultdict

from scr.geometry.contours.extraction import extract_frame_contours
from scr.geometry.contours.relations import contour_belongs_to_outer
from scr.geometry.contours.topology import build_signed_region
from scr.geometry.contours.shapes import contour_to_shape


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

    # --------------------------------------------
    # Precompute which sunspots exist in each frame
    # --------------------------------------------
    frame_to_sids = nested_defaultdict(factory=list, depth=1)

    for sid, track in outer_tracks.items():
        for t in track:
            frame_to_sids[t].append(sid)

    # --------------------------------------------
    # Process frame by frame
    # --------------------------------------------
    for t in tqdm(sorted(frame_to_sids), desc="Building sunspots (frame-wise)", unit="frame"):

        image = images[t]

        sids = frame_to_sids[t]

        # --------------------------------
        # Build union shapes for sunspots
        # --------------------------------
        union_shapes = {}

        for sid in sids:

            outer_contours = outer_tracks[sid][t]

            valid_outer = [c for c in outer_contours if len(c) >= 3]

            if not valid_outer:
                continue

            # store original contours
            sunspots[sid][outer_name][t].extend(valid_outer)

            region_shape = build_signed_region(valid_outer)

            if region_shape is None:
                continue

            union_shapes[sid] = region_shape

        if not union_shapes:
            continue

        # --------------------------------
        # Extract contours ONCE per level
        # --------------------------------
        for name, level in levels.items():

            contours = extract_frame_contours(
                image=image,
                level=level,
                min_area=min_area.get(name, 0.0),
            )

            # --------------------------------
            # Assign contours to sunspots
            # --------------------------------
            for c in contours:
                shape = contour_to_shape(c)  # convert shapes once
                if shape is None:
                    continue

                for sid, union_shape in union_shapes.items():

                    if contour_belongs_to_outer(
                            inner=shape,
                            outer=union_shape,
                            mode=containment_mode,
                            min_fraction=min_containment,
                    ):
                        sunspots[sid][name][t].append(c)

    return sunspots
