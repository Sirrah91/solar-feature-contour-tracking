from shapely import Polygon, LineString, Point, MultiPolygon, GeometryCollection
from shapely.prepared import prep, PreparedGeometry
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from scr.utils.types_alias import Contour, Contours
from scr.utils.filesystem import is_empty

from scr.geometry.contours.normalization import close_contour


def contour_to_shape(
        contour: Contour,
        holes: Contours | None = None,
        close: bool = True
) -> Polygon | LineString | Point:
    """
    Convert a contour and optional holes into a shapely shape.

    Parameters:
        contour: Nx2 array of (y, x) coordinates for the outer boundary.
        holes: Optional list of Nx2 arrays defining holes.
        close: If True, ensure contours are closed polygons.

    Returns:
        Shapely geometry (Polygon, LineString, or Point).
    """
    if close:
        contour = close_contour(contour)
        holes = [close_contour(h) for h in holes] if not is_empty(holes) else []

    if len(contour) == 1:
        return Point(contour[0])
    if len(contour) < 3:
        return LineString(contour)

    try:
        poly = Polygon(shell=contour, holes=holes)
        return poly if poly.is_valid else poly.buffer(0)
    except Exception:
        return Polygon(contour).convex_hull


def union_contours(
        contours: Contours
) -> Polygon | MultiPolygon | GeometryCollection | None:
    """
    Convert a list of contours into a single merged shapely shape,
    reusing robust single-contour logic.
    """
    if not contours:
        return None

    # Reuse the robust logic for each individual contour
    shapes = [contour_to_shape(c) for c in contours]

    # unary_union handles a list of mixed types (Polygons, Lines, Points)
    merged = unary_union(shapes)

    return None if merged.is_empty else merged


def prepare_shape(
        shape: BaseGeometry
) -> PreparedGeometry:
    """
    Prepare a Shapely geometry for repeated spatial predicates.
    """
    return prep(shape)


def contour_area_shapely(
        contour: Contour,
        hole_contours: Contours | None = None,
        close: bool = True
) -> float:
    """
    Compute the geometric area of a contour using Shapely.

    Parameters:
        contour: Nx2 array of points.
        hole_contours: List of inner hole contours.
        close: Whether to enforce closed polygons.

    Returns:
        Area of the contour polygon. Returns 0 for non-closed shapes.
    """
    return contour_to_shape(contour=contour, holes=hole_contours, close=close).area


def contour_length_shapely(
        contour: Contour,
        close: bool = True
) -> float:
    """
    Compute the geometric perimeter/length of a contour using Shapely.

    Parameters:
        contour: Nx2 array of points.
        close: Whether to close the contours.

    Returns:
        Perimeter of the shape (polygon or line).
    """
    return contour_to_shape(contour=contour, close=close).length


def total_contours_area_shapely(
        contours: Contours,
        hole_contours: list[Contours] | None = None,
        close: bool = True
) -> float:
    """
    Compute total area from outer contours, optionally subtracting holes.

    Parameters:
        contours: List of outer boundary contours.
        hole_contours: List of lists of inner hole contours for each outer contour (or None).
        close: Whether to enforce closed polygons.

    Returns:
        Total geometric area.
    """
    if hole_contours is None:
        hole_contours = [[] for _ in contours]

    # Valid list of lists
    if not (isinstance(hole_contours, list) and all(isinstance(inner, list) for inner in hole_contours)):
        raise ValueError(
            "hole_contours parameter must be a list of lists of inner hole contours for each outer contour")

    return sum(contour_area_shapely(contour=contour, hole_contours=holes, close=close)
               for contour, holes in zip(contours, hole_contours))


def total_contours_length_shapely(
        contours: Contours,
        close: bool = True
) -> float:
    """
    Compute total perimeter/length from a list of contours.

    Parameters:
        contours: List of Nx2 arrays representing contours.
        close: Whether to close the contours.

    Returns:
        Sum of the lengths of all contours.
    """
    return sum(contour_length_shapely(contour=contour, close=close) for contour in contours)


def do_contours_intersects(
        contour1: Contour,
        contour2: Contour
) -> bool:
    """
    Return True if two contours intersect.
    """
    return contour_to_shape(contour1).intersects(contour_to_shape(contour2))


def compute_iou(
        contour_shape1: Polygon | LineString | Point,
        contour_shape2: Polygon | LineString | Point,
) -> float:
    """
    Compute the Intersection-over-Union (IoU) between two contour.
    """
    if not (contour_shape1.is_valid and contour_shape2.is_valid):
        return 0.0

    if not (isinstance(contour_shape1, Polygon) and isinstance(contour_shape2, Polygon)):
        return 0.0

    intersection = contour_shape1.intersection(contour_shape2).area

    if intersection > 0.0:
        union = contour_shape1.union(contour_shape2).area
        if union > 0.0:
            return intersection / union
    return 0.0
