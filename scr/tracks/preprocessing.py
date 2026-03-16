from scr.utils.types_alias import Tracks, Events
from scr.tracks.filtering import filter_tracks_by_lifetime
from scr.tracks.nesting import collapse_nested_tracks, remove_nested_contours_per_frame
from scr.tracks.events import prune_events_after_filtering
from scr.geometry.contours.filtering import filter_contours_by_vertices


def preprocess_tracks(
        tracks: Tracks,
        filtering: dict | None,
        *,
        events: Events | None = None,
        min_containment: float = 0.8,
) -> tuple[Tracks, Events | None]:
    """
    Apply optional contour-vertex filtering, lifetime filtering, nesting logic,
    and optional clockwise filtering to a dictionary of tracks.
    Optionally prune events to remaining track IDs.

    Parameters
    ----------
    tracks : dict
        Mapping track_id -> track data.
    filtering : dict, optional
        Supported keys:
            - min_frames : int
            - remove_nested : bool
            - collapse_nested : bool
            - min_vertices : int
            - max_healing_gap : float
            - max_closing_gap : float
    events : list, optional
        Event list to be pruned after filtering.
    min_containment : float
        Containment threshold for nesting operations.

    Returns
    -------
    filtered_tracks : dict
    filtered_events : list or None
    """

    if not filtering:
        return tracks, events

    mode_remove = filtering.get("remove_nested", False)
    mode_collapse = filtering.get("collapse_nested", False)

    if mode_remove and mode_collapse:
        raise ValueError(
            "remove_nested and collapse_nested are mutually exclusive."
        )

    # -----------------------
    # 0) Optional contour vertex filtering
    # -----------------------
    min_vertices = filtering.get("min_vertices")
    max_healing_gap = filtering.get("max_healing_gap")
    max_closing_gap = filtering.get("max_closing_gap")

    if min_vertices is not None or max_healing_gap is not None or max_closing_gap is not None:
        for tid, frames in tracks.items():
            for frame, contours in frames.items():
                frames[frame] = filter_contours_by_vertices(
                    contours,
                    min_vertices=min_vertices or 4,
                    max_healing_gap=max_healing_gap or 0.0,
                    max_closing_gap=max_closing_gap or 0.0,
                )

    # -----------------------
    # 1) Nesting logic
    # -----------------------
    if mode_remove:
        tracks = remove_nested_contours_per_frame(
            tracks=tracks,
            mode="covers",
            min_containment=min_containment,
        )

    elif mode_collapse:
        tracks, _ = collapse_nested_tracks(
            tracks,
            mode="covers",
            min_containment=min_containment,
            min_overlap_frames=2,
            min_nesting_fraction=0.5,
        )

    # -----------------------
    # 2) Lifetime filtering
    # -----------------------
    if "min_frames" in filtering:
        tracks = filter_tracks_by_lifetime(
            tracks,
            min_lifetime=filtering["min_frames"],
        )

    # -----------------------
    # 3) Optional event pruning
    # -----------------------
    if events is not None:
        events = prune_events_after_filtering(
            events=events,
            remaining_ids=set(tracks.keys()),
        )

    return tracks, events
