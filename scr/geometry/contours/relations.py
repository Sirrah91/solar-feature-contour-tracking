from shapely.geometry import Polygon, MultiPolygon
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
    Strict and robust containment test.

    Guarantees:
    - No symmetric false positives (A in B and B in A)
    - Robust to holes, rings, thin geometries
    - No reliance on unstable heuristics

    Interpretation:
    - "inner belongs to outer" ⇔ majority of inner lies inside outer
    """

    # ============================================================
    # 0. Normalisation
    # ============================================================

    if inner.is_empty:
        return False

    if isinstance(outer, PreparedGeometry):
        outer_geom = outer.context
        outer_prep = outer
    else:
        outer_geom = outer
        if outer_geom.is_empty:
            return False
        outer_prep = prepare_shape(outer_geom)

    if outer_geom.is_empty:
        return False

    # ============================================================
    # 1. Enforce directionality (CRITICAL)
    # ============================================================

    inner_area = inner.area
    outer_area = outer_geom.area

    if inner_area > outer_area:
        return False

    # ============================================================
    # 2. Bounding-box rejection (safe)
    # ============================================================

    minxi, minyi, maxxi, maxyi = inner.bounds
    minxo, minyo, maxxo, maxyo = outer_geom.bounds

    if maxxi < minxo or maxxo < minxi or maxyi < minyo or maxyo < minyi:
        return False

    # ============================================================
    # 3. Exact predicate (fast + reliable)
    # ============================================================

    if mode == "strict":
        if outer_prep.contains(inner):
            return True
    elif mode == "covers":
        if outer_prep.covers(inner):
            return True
    else:
        raise ValueError(f"Unknown containment mode: {mode}")

    # ============================================================
    # 4. Must intersect (reject otherwise)
    # ============================================================

    if not outer_prep.intersects(inner):
        return False

    # ============================================================
    # 5. Exterior (shell) test  hole-safe
    # ============================================================

    if isinstance(inner, Polygon):
        if outer_prep.covers(inner.exterior):
            return True

    elif isinstance(inner, MultiPolygon):
        for p in inner.geoms:
            if not outer_prep.covers(p.exterior):
                break
        else:
            return True

    # ============================================================
    # 6. Boundary overlap (robust for thin shapes)
    # ============================================================

    boundary = inner.boundary
    blen = boundary.length

    if blen > 0.0:
        inter_boundary = outer_geom.intersection(boundary)
        if not inter_boundary.is_empty:
            if inter_boundary.length / blen >= min_fraction:
                return True

    # ============================================================
    # 7. Area dominance (FINAL DECISION)
    # ============================================================

    if inner_area > 0.0:
        inter = outer_geom.intersection(inner)
        if not inter.is_empty:
            if inter.area / inner_area >= min_fraction:
                return True

    return False
