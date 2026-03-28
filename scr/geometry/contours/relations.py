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

    Strategy
    --------
    1. Fast bounding-box rejection
    2. Fast exact predicate (contains / covers)
    3. Robust fallback using area fraction

    Notes
    -----
    - `outer` may be PreparedGeometry
    - `inner` must be raw geometry
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

    # --- Fast bounding-box rejection ---
    minxi, minyi, maxxi, maxyi = inner.bounds
    minxo, minyo, maxxo, maxyo = outer_geom.bounds

    if (
            minxi < minxo or
            minyi < minyo or
            maxxi > maxxo or
            maxyi > maxyo
    ):
        return False

    # --- Fast exact test ---
    if mode == "strict":
        exact = outer_prep.contains(inner)

    elif mode == "covers":
        exact = outer_prep.covers(inner)

    else:
        raise ValueError(f"Unknown containment mode: {mode}")

    if exact:
        return True

    # --- Robust fallback ---
    if not outer_prep.intersects(inner):
        return False

    inter = outer_geom.intersection(inner)

    if inter.is_empty:
        return False

    if inner.area == 0:
        return False

    return inter.area / inner.area >= min_fraction
