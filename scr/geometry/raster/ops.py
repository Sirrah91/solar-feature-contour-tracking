import numpy as np
from scr.utils.types_alias import Mask
from scr.utils.filesystem import is_empty


def subtract(
        mask1: Mask,
        mask2: Mask
) -> Mask:
    """
    Fractional set difference.
    """
    return np.clip(mask1 - mask2, 0.0, 1.0)


def union(
        mask1: Mask,
        mask2: Mask
) -> Mask:
    """
    Fractional union.
    """
    return np.clip(mask1 + mask2, 0.0, 1.0)


def intersection(
        mask1: Mask,
        mask2: Mask
) -> Mask:
    """
    Fractional intersection.
    """
    return np.minimum(mask1, mask2)


def containment_ratio(
        mask_small: Mask,
        mask_large: Mask
) -> float:
    """
    Fraction of mask_small contained in mask_large.

    Works for both boolean and fractional masks.
    """
    common = intersection(mask_small, mask_large).sum()
    area_small = mask_small.sum()

    if area_small <= 0.0:
        return 0.0

    return float(common) / float(area_small)
