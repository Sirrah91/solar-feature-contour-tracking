from shapely.geometry.base import BaseGeometry
from shapely.prepared import PreparedGeometry
from shapely import Polygon, MultiPolygon

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

    Philosophy
    ----------
    - Topology-aware (holes, rings)
    - Robust to thin / degenerate shapes
    - Cheap tests first, expensive last

    Decision pipeline
    -----------------
    REJECT:
        1. Empty geometry
        2. Bounding-box disjoint

    ACCEPT (cheap & robust):
        3. Exact predicate (covers / contains)
        4. Representative point inside
        5. Exterior (shell) inside

    REFINE (expensive):
        6. Intersection existence
        7. Boundary overlap fraction
        8. Area overlap fraction (final fallback)
    """

    # ============================================================
    # 0. Normalisation & preparation
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
    # 1. Fast rejection (bounding boxes)
    # ============================================================

    minxi, minyi, maxxi, maxyi = inner.bounds
    minxo, minyo, maxxo, maxyo = outer_geom.bounds

    if maxxi < minxo or maxxo < minxi or maxyi < minyo or maxyo < minyi:
        return False

    # ============================================================
    # 2. Exact predicate
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
    # 3. Topology-aware cheap acceptance
    # ============================================================

    # 3a. Representative point (hole-safe)
    rp = inner.representative_point()
    if outer_prep.covers(rp):
        return True

    # 3b. Exterior shells (ring-safe)
    if isinstance(inner, Polygon):
        if outer_prep.covers(inner.exterior):
            return True

    elif isinstance(inner, MultiPolygon):
        if all(outer_prep.covers(p.exterior) for p in inner.geoms):
            return True

    # ============================================================
    # 4. Intersection existence (cheap-ish rejection)
    # ============================================================

    if not outer_prep.intersects(inner):
        return False

    # ============================================================
    # 5. Boundary-based refinement (thin-shape robust)
    # ============================================================

    boundary = inner.boundary
    blen = boundary.length

    if blen > 0:
        inter_boundary = outer_geom.intersection(boundary)
        if not inter_boundary.is_empty:
            if inter_boundary.length / blen >= min_fraction:
                return True

    # ============================================================
    # 6. Area-based fallback (last resort)
    # ============================================================

    area = inner.area
    if area == 0:
        return False

    inter_area = outer_geom.intersection(inner)

    if inter_area.is_empty:
        return False

    return (inter_area.area / area) >= min_fraction
