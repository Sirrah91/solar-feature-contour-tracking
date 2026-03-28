from scr.utils.types_alias import SunspotPhases, SunspotPart
from scr.utils.filesystem import is_empty

from scr.geometry.crop.bounds import compute_crop_bounds


def common_crop_shape_from_tracks(
        sunspot_phases: SunspotPhases,
        *,
        key_region: SunspotPart,
        margin: int = 0,
) -> tuple[int, int]:
    """
    Compute a common (height, width) covering all contours
    of a selected region across all phases and frames.
    """
    heights: list[int] = []
    widths: list[int] = []

    for phase_data in sunspot_phases.values():
        track = phase_data.get(key_region)
        if track is None:
            continue

        for contours in track.values():
            if is_empty(contours):
                continue

            y_min, y_max, x_min, x_max = compute_crop_bounds(
                contours,
                margin=margin,
            )

            heights.append(y_max - y_min)
            widths.append(x_max - x_min)

    if not heights:
        raise ValueError("No valid contours found")

    return max(heights), max(widths)
