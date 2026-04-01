from scr.utils.types_alias import Contours
from scr.utils.filesystem import is_empty


def filter_empty_contours(
        contours: Contours
) -> Contours | None:
    """
    Remove None entries from contour list, return None if all are None.

    Parameters:
        contours: List of contours or None.

    Returns:
        Cleaned list of contours or None.
    """
    cleaned = [c for c in contours if c is not None]
    return None if is_empty(cleaned) else cleaned
