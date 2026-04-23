import numpy as np
from shapely.strtree import STRtree

from scr.utils.types_alias import Contour, Contours, Mask
from scr.utils.filesystem import is_empty

from scr.geometry.contours.extraction import find_contours
from scr.geometry.contours.distance import contours_distance
from scr.geometry.contours.normalization import normalize_contour_input
from scr.geometry.contours.shapes import contour_to_shape, compute_iou
from scr.geometry.raster.api import rasterize

from scr.morphology.binary import expand_mask


def select_support_contours(
        primary_contours: Contours,
        support_contours: Contours,
        min_iou: float = 0.2,
) -> Contours:
    """
    For each primary contour, select the best matching support contour.

    Matching rule:
    - support centroid must lie inside the primary contour
    - if multiple candidates exist, pick the largest one

    Returns
    -------
    list of contours
        One support contour per primary contour (if found).
    """
    if is_empty(primary_contours) or is_empty(support_contours):
        return []

    # 1. Convert all support contours to shapes once
    support_shapes = [contour_to_shape(s) for s in support_contours]

    # 2. Build the spatial index
    tree = STRtree(support_shapes)

    selected: list = []

    for primary in primary_contours:
        p_shape = contour_to_shape(primary)

        # 3. Query the tree: "Which support shapes have bounding boxes that touch me?"
        # indices is an array of integers pointing to support_shapes
        indices = tree.query(p_shape, predicate="intersects")

        best_support = None
        max_iou = 0.0

        # 4. Only run the expensive IoU math on viable candidates
        for idx in indices:
            s_shape = support_shapes[idx]
            iou = compute_iou(p_shape, s_shape)

            if iou > max_iou:
                max_iou = iou
                best_support = support_contours[idx]

        if best_support is not None and max_iou >= min_iou:
            selected.append(best_support)

    return selected


def select_best_contour(
        input_contour: Contour,
        candidates: Contours,
        erosion_image_shape: tuple[int, int],
        expansion_pixels: int
) -> tuple[Contour, Mask] | None:
    """
    From candidate contours, select the one best overlapping the input.

    Returns:
        (best_contour, best_mask) or None.
    """
    input_poly = contour_to_shape(np.reshape(input_contour, (-1, 2)))

    max_area = 0
    best_contour = None
    best_mask = None

    closest_idx = np.argmin([
        contours_distance(input_contour, cnt) for cnt in candidates
    ]) if not is_empty(candidates) else 0

    for idx, cnt in enumerate(candidates):
        cnt = np.reshape(cnt, (-1, 2))
        mask = rasterize(
            contours=normalize_contour_input(cnt),
            shape=erosion_image_shape,
            mode="surface",
            engine="skimage",
            use_orientation=False,
            dtype=bool
        )
        mask = expand_mask(mask, expansion_pixels=expansion_pixels)

        cnt_found = find_contours(image=mask, level=0.5)
        if is_empty(cnt_found):
            continue

        cnt_poly = contour_to_shape(cnt_found[0])

        if cnt_poly.intersects(input_poly):
            area = cnt_poly.intersection(input_poly).area
        else:
            area = 0.0

        if area > max_area or (max_area == 0. and idx == closest_idx):
            max_area = area
            best_contour = cnt_found[0]
            best_mask = mask

    if best_contour is None:
        return None

    return best_contour, best_mask
