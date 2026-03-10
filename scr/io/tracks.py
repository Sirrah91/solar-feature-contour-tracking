from scr.utils.types_alias import StatsByQuantity, Events, Metadata
from scr.utils.filesystem import check_dir

from scr.io.npz import load_npz, save_npz


def load_track_file(
        filename: str
) -> tuple[dict, StatsByQuantity, Metadata, Events]:
    """
    Load track data, statistics, and metadata from a .npz file.

    Parameters:
        filename: Path to the saved .npz archive.

    Returns:
        Tuple of (tracks, statistics, metadata, events) dictionaries.
    """
    data = load_npz(filename)
    return (
        data["tracks"].item(),
        data["stats"].item(),
        data["metadata"].item(),
        list(data["events"])
    )


def save_track_file(
        filename: str,
        tracks: dict,
        stats: StatsByQuantity | None = None,
        metadata: Metadata | None = None,
        events: Events | None = None,
        **kwargs,
) -> None:
    """
    Save track data, statistics, and optional metadata to a compressed .npz file.

    Parameters:
        filename: File path to save the .npz archive.
        tracks: Dictionary of tracked contours.
        stats: Optional dictionary of statistics per track.
        metadata: Optional additional information (e.g., parameters).
        events: Optional information on merging/splitting.
    """
    check_dir(filename, is_file=True)

    save_npz(
        filename,
        tracks=tracks,
        stats=stats or {},
        metadata=metadata or {},
        events=events or [],
        **kwargs,
    )
