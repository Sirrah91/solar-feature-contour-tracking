from skimage.morphology import dilation, erosion, footprint_rectangle

from scr.utils.types_alias import Mask


def dilate_mask(
        mask: Mask,
        dilate_pixels: int = 0
) -> Mask:
    """Dilate a binary mask using a square structuring element."""
    if dilate_pixels <= 0:
        return mask
    return dilation(mask, footprint_rectangle((2 * dilate_pixels + 1, 2 * dilate_pixels + 1)))


def erode_mask(
        mask: Mask,
        erode_pixels: int = 0
) -> Mask:
    """Erode a binary mask using a square structuring element."""
    if erode_pixels <= 0:
        return mask
    return erosion(mask, footprint_rectangle((2 * erode_pixels + 1, 2 * erode_pixels + 1)))
