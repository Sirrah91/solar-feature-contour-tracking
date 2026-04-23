import numpy as np

from scr.utils.types_alias import Mask

from scr.morphology.ops import erode_mask, dilate_mask


def expand_mask(
        mask: Mask,
        expansion_pixels: int = 0,
        border_only: bool = False
) -> Mask:
    """
    Dilate or erode a binary mask, with optional border-only extraction.

    Parameters:
        mask: Binary or float mask.
        expansion_pixels: Positive to dilate, negative to erode.
        border_only: If True, return only the edge pixels after expansion.

    Returns:
        Modified binary mask.
    """
    if expansion_pixels > 0:
        mask = dilate_mask(mask, dilate_pixels=expansion_pixels)
    elif expansion_pixels < 0:
        mask = erode_mask(mask, erode_pixels=-expansion_pixels)

    if border_only:
        mask_bool = mask.astype(bool)
        border = mask_bool & ~erode_mask(mask_bool, erode_pixels=1)
        mask = border.astype(mask.dtype)

    return mask


def threshold_and_erode(
        image: np.ndarray,
        threshold: float,
        iterations: int
) -> Mask:
    """Threshold image and optionally erode."""
    thresh = np.abs(image) > threshold
    if iterations > 0:
        return erode_mask(thresh, erode_pixels=iterations)
    return thresh
