import numpy as np

from scr.utils.types_alias import Tracks

from scr.geometry.contours.orientation import is_ccw


def filter_tracks_by_lifetime(
        tracks: Tracks,
        min_lifetime: int = 0,
        max_lifetime: int = np.inf
) -> Tracks:
    if min_lifetime <= 0 and max_lifetime == np.inf:
        return tracks
    return {tid: hist for tid, hist in tracks.items() if min_lifetime <= len(hist) <= max_lifetime}


def remove_clockwise_contours(
        tracks: Tracks
) -> Tracks:
    """
    Remove all contours with negative signed area (CW orientation).
    No geometric checks, purely by contour orientation.

    Parameters
    ----------
    tracks : dict
        {track_id: {frame_idx: [contours]}}

    Returns
    -------
    cleaned_tracks : dict
        Tracks with negative-area contours removed.
    """
    cleaned = {}

    for tid, frames in tracks.items():
        new_frames = {}

        for t, contours in frames.items():
            # Keep only contours whose signed area is >= 0
            kept = [c for c in contours if is_ccw(c)]

            if kept:  # keep frame only if it still has contours
                new_frames[t] = kept

        if new_frames:  # keep track only if not empty
            cleaned[tid] = new_frames

    return cleaned
