import numpy as np

from scr.config.numerics import WP
from scr.utils.filesystem import is_empty
from scr.utils.types_alias import Contours, Mask, Masks
from scr.geometry.contours.shapes import contour_to_shape
from scr.geometry.raster.fill import contours_to_signed_fractional_mask
from scr.geometry.raster.border import contours_to_border_mask
from scr.geometry.raster.ops import subtract


def safe_subtract(
        mask1: Mask,
        mask2: Mask,
) -> Mask:
    if is_empty(mask1) | is_empty(mask2):
        return mask1
    return subtract(mask1, mask2)


def compute_masks(
        contours: Contours,
        shape: tuple[int, int]
) -> tuple[Masks, Masks]:
    """
    Compute filling-factor and border masks for a list of contours.
    """
    if is_empty(contours):
        return [], []

    fill_masks = [contours_to_signed_fractional_mask(c, shape) for c in contours]
    border_masks = [contours_to_border_mask(c, shape) for c in contours]

    return fill_masks, border_masks


def compute_nesting_matrix(contours: Contours) -> np.ndarray:
    """
    Compute contour nesting relationships.

    matrix[child, parent] = True
    """

    if is_empty(contours):
        return np.zeros((0, 0), dtype=bool)

    polygons = [contour_to_shape(c) for c in contours]

    num = len(polygons)
    nesting = np.zeros((num, num), dtype=bool)

    for i in range(num):
        for j in range(num):
            if i != j and polygons[j].contains(polygons[i]):
                nesting[i, j] = True

    return nesting


def build_refined_masks(
        *,
        contours: Contours,
        individual_masks: Masks | None = None,
        shape: tuple[int, int] | None = None,
        nesting_matrix: np.ndarray | None = None,
) -> Masks:
    """
    Build refined masks where holes are removed from parent contours.
    """

    if is_empty(contours):
        return []

    if individual_masks is None:
        if shape is None:
            raise ValueError("shape must be provided when masks are not given")

        individual_masks = [
            contours_to_signed_fractional_mask(c, shape)
            for c in contours
        ]

    if nesting_matrix is None:
        nesting_matrix = compute_nesting_matrix(contours)

    num = len(individual_masks)

    # --------------------------------------------------
    # find direct parent of each contour
    # --------------------------------------------------

    parents = [None] * num

    for child in range(num):

        candidates = np.where(nesting_matrix[child])[0]

        if len(candidates) == 0:
            continue

        parent = candidates[0]

        for p in candidates[1:]:
            if nesting_matrix[p, parent]:
                parent = p

        parents[child] = parent

    # --------------------------------------------------
    # build children list
    # --------------------------------------------------

    children = [[] for _ in range(num)]

    for child, parent in enumerate(parents):
        if parent is not None:
            children[parent].append(child)

    # --------------------------------------------------
    # build refined masks
    # --------------------------------------------------

    refined_masks = []

    for i in range(num):

        # only positive contours define regions
        if np.max(individual_masks[i]) <= 0:
            continue

        m = individual_masks[i].copy()

        for child in children[i]:

            # subtract holes
            if np.min(individual_masks[child]) < 0:
                m = safe_subtract(m, np.abs(individual_masks[child]))

        refined_masks.append(m)

    return refined_masks


def overall_mask(
        masks: Masks,
        dtype: type = np.float64
) -> Mask:
    """Combine a list of masks into a single clipped mask."""
    if is_empty(masks):
        return np.asarray([], dtype=dtype)

    stacked = np.nansum(np.stack(masks, axis=0), axis=0)
    return np.clip(stacked, 0.0, 1.0).astype(dtype)


def corr_mask(
        mask: Mask,
        weights: np.ndarray
) -> Mask:
    """Multiply mask by weights, ignoring NaNs and zeros."""
    corrected_mask = np.zeros_like(mask, dtype=float)
    valid = np.isfinite(mask) & np.isfinite(weights)
    corrected_mask[valid] = mask[valid] * weights[valid]
    return corrected_mask
