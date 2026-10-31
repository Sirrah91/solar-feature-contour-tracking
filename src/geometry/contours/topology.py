from shapely.prepared import PreparedGeometry
from src.utils.types_alias import Contours
from src.geometry.contours.area import contour_area
from src.geometry.contours.orientation import is_ccw
from src.geometry.contours.shapes import contour_to_shape, prepare_shape


def build_signed_region(contours: Contours) -> PreparedGeometry | None:
    """
    Build a signed region from contours.

    Orientation determines topology:

        CCW → add region
        CW  → subtract region

    Produces:

        outer − holes + islands − ponds ...
    """

    region = None

    contours = sorted(contours, key=lambda c: abs(contour_area(c)), reverse=True)

    for c in contours:

        if len(c) < 3:
            continue

        shape = contour_to_shape(c)

        if shape is None or shape.is_empty:
            continue

        if is_ccw(c):
            region = shape if region is None else region.union(shape)
        else:
            if region is not None:
                region = region.difference(shape)

    if region is None:
        return None

    return prepare_shape(region)
