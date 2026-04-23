from scr.utils.types_alias import Contour, Contours

from scr.geometry.contours.area import contour_signed_area


def is_ccw(
        contour: Contour
) -> bool:
    """Return True if contour is oriented CCW."""
    return contour_signed_area(contour) > 0.


def classify_contours(
        contours: Contours
) -> tuple[Contours, Contours]:
    outer = []
    holes = []

    for c in contours:
        if is_ccw(c):
            outer.append(c)  # CCW → outer boundary
        else:
            holes.append(c)  # CW → inner hole

    return outer, holes
