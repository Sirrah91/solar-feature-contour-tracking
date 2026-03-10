from tqdm import tqdm

from scr.utils.types_alias import Sunspots, TracksLabeled, AssociationMode
from scr.utils.collections import nested_defaultdict
from scr.utils.dict import separate_key_value

from scr.geometry.contours.shapes import contour_to_shape, contours_to_shape
from scr.geometry.contours.relations import contour_belongs_to_outer


# ============================================================
# Public API
# ============================================================

def associate_levels_flat(
        all_tracks: list[TracksLabeled],
        mode: AssociationMode = "covers",
        min_fraction: float = 0.8,
) -> Sunspots:
    """
    Sequentially associate contour levels from outermost inward.

    Rules:
    - Inner exists only if its direct parent exists.
    - No structural nesting (flat dictionary).
    - Each inner contour attaches to at most one parent per frame.
    """
    if not all_tracks:
        return {}

    # ----------------------------------------------------------
    # Initialize from outermost
    # ----------------------------------------------------------

    sunspots = _initialize_from_outer(all_tracks[0])
    parent_label, _ = separate_key_value(all_tracks[0])

    # ----------------------------------------------------------
    # Merge deeper levels iteratively
    # ----------------------------------------------------------

    for tracks_labeled in all_tracks[1:]:
        sunspots = _merge_one_level_optimized(
            sunspots=sunspots,
            tracks_labeled=tracks_labeled,
            parent_label=parent_label,
            mode=mode,
            min_fraction=min_fraction,
        )

        parent_label, _ = separate_key_value(tracks_labeled)

    return sunspots


# ============================================================
# Internal helpers
# ============================================================

def _initialize_from_outer(
        tracks_labeled:
        TracksLabeled
) -> Sunspots:
    label, tracks = separate_key_value(tracks_labeled)

    return {
        track_id: {label: data}
        for track_id, data in tracks.items()
    }


def _merge_one_level_optimized(
        sunspots: Sunspots,
        tracks_labeled: TracksLabeled,
        parent_label: str,
        mode: AssociationMode,
        min_fraction: float,
) -> Sunspots:
    """
    Optimized merging:
    - Pre-index inner tracks by frame
    - Precompute parent union shapes per frame
    - Ensure each inner contour attaches only once
    """

    new_label, new_tracks = separate_key_value(tracks_labeled)

    # ----------------------------------------------------------
    # Build frame index for inner contours
    # frame_index[t] = [(track_id, contour), ...]
    # ----------------------------------------------------------

    frame_index = nested_defaultdict(factory=list)

    for inner_id, inner_data in new_tracks.items():
        for t, contours in inner_data.items():
            for c in contours:
                frame_index[t].append((inner_id, c))

    # Track which inner contours were already assigned
    assigned = nested_defaultdict(factory=set)
    # assigned[t] = set(id(contour))

    # ----------------------------------------------------------
    # Iterate over sunspots
    # ----------------------------------------------------------

    for spot_id, parts in tqdm(sunspots.items(), desc=f"Associating {new_label} tracks to sunspots", unit="sunspot"):

        if parent_label not in parts:
            continue

        parent_data = parts[parent_label]

        # Ensure new label key exists
        if new_label not in parts:
            parts[new_label] = nested_defaultdict(factory=list)

        # ------------------------------------------------------
        # Frame-wise matching
        # ------------------------------------------------------

        for t, parent_contours in parent_data.items():

            if t not in frame_index:
                continue

            # Precompute union once
            union_shape = contours_to_shape(parent_contours)

            for inner_id, contour in frame_index[t]:

                contour_id = id(contour)

                # Skip if already attached to another spot
                if contour_id in assigned[t]:
                    continue

                if contour_belongs_to_outer(
                        inner=contour_to_shape(contour),
                        outer=union_shape,
                        mode=mode,
                        min_fraction=min_fraction,
                ):
                    parts[new_label][t].append(contour)
                    assigned[t].add(contour_id)

    return sunspots
