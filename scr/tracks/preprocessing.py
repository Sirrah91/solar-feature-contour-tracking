from scr.utils.types_alias import Tracks, Events

from scr.tracks.filtering import (
    filter_tracks_by_lifetime,
)
from scr.tracks.nesting import (
    collapse_nested_tracks,
    remove_nested_contours_per_frame,
)
from scr.tracks.events import prune_events_after_filtering


def preprocess_tracks(
        tracks: Tracks,
        filtering: dict | None,
        *,
        events: Events | None = None,
        min_containment: float = 0.8,
) -> tuple[Tracks, Events | None]:
    """
    Apply lifetime filtering, nesting logic, and optional clockwise filtering
    to a dictionary of tracks. Optionally prune events to remaining track IDs.

    Parameters
    ----------
    tracks : dict
        Mapping track_id -> track data.
    filtering : dict, optional
        Supported keys:
            - min_frames : int
            - remove_nested : bool
            - collapse_nested : bool
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

    # 1) Lifetime filtering
    if "min_frames" in filtering:
        tracks = filter_tracks_by_lifetime(
            tracks,
            min_lifetime=filtering["min_frames"],
        )

    # 2) Nesting logic
    # Remove nested contours frame-wise (aggressive but preserves tracks)
    if mode_remove:
        tracks = remove_nested_contours_per_frame(
            tracks=tracks,
            mode="covers",
            min_containment=min_containment,
        )

    # Collapse nested tracks cautiously
    elif mode_collapse:
        tracks, _ = collapse_nested_tracks(
            tracks,
            mode="covers",
            min_containment=min_containment,
            min_overlap_frames=2,  # require at least 2 overlapping frames
            min_nesting_fraction=0.5,  # require at least 50% of overlapping frames
        )

    # 3) Optional event pruning
    if events is not None:
        events = prune_events_after_filtering(
            events=events,
            remaining_ids=set(tracks.keys()),
        )

    return tracks, events
