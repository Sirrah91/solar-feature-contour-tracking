from shapely.geometry.base import BaseGeometry
from shapely.prepared import PreparedGeometry

from scr.utils.types_alias import AssociationMode
from scr.geometry.contours.shapes import prepare_shape


def contour_belongs_to_outer(
        *,
        outer: BaseGeometry | PreparedGeometry,
        inner: BaseGeometry,
        mode: AssociationMode = "covers",
        min_fraction: float = 0.8,
) -> bool:
    """
    Decide whether `inner` contour belongs to `outer`.

    Notes
    -----
    - `outer` may be a PreparedGeometry or a raw geometry
    - `inner` MUST be a raw geometry
    """

    if inner.is_empty:
        return False

    # Ensure we have both representations
    if isinstance(outer, PreparedGeometry):
        outer_prep = outer
        outer_geom = outer.context
    else:
        outer_geom = outer
        outer_prep = prepare_shape(outer)

    if outer_geom.is_empty:
        return False

    if mode == "strict":
        return outer_prep.contains(inner)

    if mode == "covers":
        return outer_prep.covers(inner)

    if mode == "robust":
        # Fast rejection
        if not outer_prep.intersects(inner):
            return False

        # Exact test
        inter = outer_geom.intersection(inner)
        if inter.is_empty:
            return False

        # Guard against degenerate geometries
        if inner.area == 0:
            return False

        return inter.area / inner.area >= min_fraction

    raise ValueError(f"Unknown containment mode: {mode}")
