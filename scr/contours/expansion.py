import numpy as np

from scr.utils.types_alias import Contour, Mask
from scr.utils.filesystem import is_empty

from scr.geometry.contours.extraction import find_contours
from scr.geometry.contours.filtering import filter_candidate_contours

from scr.morphology.binary import threshold_and_erode

from scr.contours.selection import select_best_contour


def extract_expanded_contour(
        image: np.ndarray,
        contour: Contour,
        expansion_threshold: float = 500.0,
        iterations: int = 1,
        fill_positive: float | bool = 1.0,
        fill_negative: float | bool = 0.0,
        return_mask: bool = False
) -> Contour | tuple[Contour, Mask] | None:
    """
    Expand a contour by thresholding and morphological operations.

    Parameters:
        image: Input image.
        contour: Original contour points.
        expansion_threshold: Intensity threshold for expansion.
        iterations: Number of erosion iterations before expansion.
        fill_positive: Fill value inside the mask.
        fill_negative: Fill value outside the mask.
        return_mask: Whether to return the binary mask along with the contour.

    Returns:
        Expanded contour (and mask if requested), or None if no suitable contour found.
    """
    input_contour = np.reshape(contour, (-1, 2))

    # Step 1: Threshold and erode
    eroded_image = threshold_and_erode(image, expansion_threshold, iterations)

    # Step 2: Find candidate contours
    candidates = find_contours(eroded_image, level=0.5)
    if is_empty(candidates):
        return None

    # Step 3: Filter by distance
    candidates = filter_candidate_contours(
        input_contour,
        candidates,
        max_distance=2.0 * max(iterations, 1)
    )
    if is_empty(candidates):
        return None

    # Step 4: Select best candidate
    selected = select_best_contour(
        input_contour,
        candidates,
        erosion_image_shape=eroded_image.shape,
        expansion_pixels=iterations
    )
    if selected is None:
        return None

    best_contour, best_mask = selected

    # Step 5: Prepare mask
    best_mask = np.where(best_mask > 0, fill_positive, fill_negative)

    if return_mask:
        return best_contour, best_mask
    else:
        return best_contour
