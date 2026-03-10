from itertools import combinations
from typing import Callable

from shapely.geometry import Polygon

from scr.utils.types_alias import Tracks, TrackID, AssociationMode
from scr.utils.collections import nested_defaultdict
from scr.geometry.contours.relations import contour_belongs_to_outer
from scr.geometry.contours.shapes import contours_to_shape


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def build_shape_cache(tracks: Tracks) -> dict[tuple[TrackID, int], Polygon]:
    """Cache shapes for (track, frame)."""
    cache = {}

    for tid, frames in tracks.items():
        for frame, contours in frames.items():
            shape = contours_to_shape(contours)
            if shape is not None:
                cache[(tid, frame)] = shape

    return cache


# ---------------------------------------------------------------------
# Frame indexing
# ---------------------------------------------------------------------

def build_frame_index(tracks: Tracks) -> dict[int, list[TrackID]]:
    """Map frame → track IDs."""
    frame_index = nested_defaultdict(factory=list)

    for tid, frames in tracks.items():
        for frame in frames:
            frame_index[frame].append(tid)

    return frame_index


# ---------------------------------------------------------------------
# Geometry relation
# ---------------------------------------------------------------------

def nesting_relation(
        shape_a,
        shape_b,
        mode: AssociationMode,
        min_containment: float
) -> int:
    """
    Returns
    -------
    1  -> B inside A
    -1 -> A inside B
    0  -> no nesting
    """

    if contour_belongs_to_outer(
            outer=shape_a,
            inner=shape_b,
            mode=mode,
            min_fraction=min_containment,
    ):
        return 1

    if contour_belongs_to_outer(
            outer=shape_b,
            inner=shape_a,
            mode=mode,
            min_fraction=min_containment,
    ):
        return -1

    return 0


# ---------------------------------------------------------------------
# Pair traversal engine
# ---------------------------------------------------------------------

def process_frame_pairs(
        tracks: Tracks,
        *,
        mode: AssociationMode,
        min_containment: float,
        on_relation: Callable[[int, TrackID, TrackID, int], bool],
) -> None:
    """
    Generic engine for processing track pairs per frame.

    The callback receives:
        on_relation(frame, tid_a, tid_b, relation)

    relation values:
        1  -> B inside A
        -1 -> A inside B
        0  -> no nesting
    """

    shape_cache = build_shape_cache(tracks)
    frame_index = build_frame_index(tracks)

    for frame, tids in frame_index.items():
        for tid_a, tid_b in combinations(tids, 2):

            shape_a = shape_cache.get((tid_a, frame))
            shape_b = shape_cache.get((tid_b, frame))

            if shape_a is None or shape_b is None:
                continue

            relation = nesting_relation(
                shape_a,
                shape_b,
                mode,
                min_containment,
            )

            stop = on_relation(frame, tid_a, tid_b, relation)

            if stop:
                return


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def flatten_hierarchy(nesting: dict[TrackID, TrackID]) -> dict[TrackID, TrackID]:
    """Resolve parent chains to root parents."""

    def root(tid: TrackID) -> TrackID:
        while tid in nesting:
            tid = nesting[tid]
        return tid

    for child in list(nesting):
        nesting[child] = root(child)

    return nesting


# ---------------------------------------------------------------------
# Compute nesting map
# ---------------------------------------------------------------------

def compute_track_nesting(
        tracks: Tracks,
        *,
        mode: AssociationMode = "covers",
        min_containment: float = 0.8,
        min_overlap_frames: int = 2,
        min_nesting_fraction: float = 0.5,
) -> dict[TrackID, TrackID]:
    nesting_counts = nested_defaultdict(factory=int)
    overlap_counts = nested_defaultdict(factory=int)

    def on_relation(frame: int, tid_a: TrackID, tid_b: TrackID, relation: int) -> bool:

        pair = tuple(sorted((tid_a, tid_b)))
        overlap_counts[pair] += 1

        if relation == 1:
            nesting_counts[(tid_b, tid_a)] += 1

        elif relation == -1:
            nesting_counts[(tid_a, tid_b)] += 1

        return False

    process_frame_pairs(
        tracks,
        mode=mode,
        min_containment=min_containment,
        on_relation=on_relation,
    )

    nesting = {}

    for (child, parent), count in nesting_counts.items():

        overlap = overlap_counts.get(tuple(sorted((child, parent))), 0)

        if overlap and count >= min_overlap_frames and count / overlap >= min_nesting_fraction:
            nesting[child] = parent

    return flatten_hierarchy(nesting)


# ---------------------------------------------------------------------
# Remove nested tracks
# ---------------------------------------------------------------------

def remove_nested_tracks(
        tracks: Tracks,
        *,
        mode: AssociationMode = "covers",
        min_containment: float = 0.8,
) -> Tracks:
    removed = set()

    def on_relation(frame: int, tid_a: TrackID, tid_b: TrackID, relation: int) -> bool:

        if relation == 1:
            removed.add(tid_b)

        elif relation == -1:
            removed.add(tid_a)

        return False

    process_frame_pairs(
        tracks,
        mode=mode,
        min_containment=min_containment,
        on_relation=on_relation,
    )

    return {
        tid: frames
        for tid, frames in tracks.items()
        if tid not in removed
    }


# ---------------------------------------------------------------------
# Collapse nested tracks
# ---------------------------------------------------------------------

def collapse_nested_tracks(
        tracks: Tracks,
        *,
        mode: AssociationMode = "covers",
        min_containment: float = 0.8,
        min_overlap_frames: int = 2,
        min_nesting_fraction: float = 0.5,
) -> tuple[Tracks, dict[TrackID, TrackID]]:
    nesting = compute_track_nesting(
        tracks,
        mode=mode,
        min_containment=min_containment,
        min_overlap_frames=min_overlap_frames,
        min_nesting_fraction=min_nesting_fraction,
    )

    collapsed = nested_defaultdict(factory=list, depth=2)

    for tid, frames in tracks.items():
        for frame, contours in frames.items():
            collapsed[tid][frame].extend(contours)

    for child, parent in nesting.items():

        if child not in collapsed or parent not in collapsed:
            continue

        for frame, contours in collapsed[child].items():
            collapsed[parent][frame].extend(contours)

        del collapsed[child]

    return collapsed, nesting


# ---------------------------------------------------------------------
# Remove nested contours per frame
# ---------------------------------------------------------------------

def remove_nested_contours_per_frame(
        tracks: Tracks,
        *,
        mode: AssociationMode = "covers",
        min_containment: float = 0.8,
) -> Tracks:
    frames_to_drop = set()

    def on_relation(frame: int, tid_a: TrackID, tid_b: TrackID, relation: int) -> bool:

        if relation == 1:
            frames_to_drop.add((tid_b, frame))

        elif relation == -1:
            frames_to_drop.add((tid_a, frame))

        return False

    process_frame_pairs(
        tracks,
        mode=mode,
        min_containment=min_containment,
        on_relation=on_relation,
    )

    cleaned = {}

    for tid, frames in tracks.items():

        new_frames = {
            frame: contours
            for frame, contours in frames.items()
            if (tid, frame) not in frames_to_drop
        }

        if new_frames:
            cleaned[tid] = new_frames

    return cleaned
