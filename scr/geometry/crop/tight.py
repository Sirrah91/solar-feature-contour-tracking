import numpy as np
from scr.utils.types_alias import Contour, Contours
from scr.geometry.crop.bounds import compute_crop_bounds


def crop_tight(
        image: np.ndarray,
        contours: Contour | Contours,
        *,
        margin: int = 0,
        return_offsets: bool = True,
) -> np.ndarray | tuple[np.ndarray, tuple[int, int]]:
    """
    Crop image tightly around contours with an optional margin.

    Parameters
    ----------
    image : ndarray
        2D image.
    contours : list of contours
        Contours defining the object extent.
    margin : int
        Extra pixels added on each side.
    return_offsets : bool
        Return image offset to transform contours

    Returns
    -------
    cropped_image : ndarray
    (y_offset, x_offset) : tuple of int
        Offset of the crop relative to the original image.
    """
    y_min, y_max, x_min, x_max = compute_crop_bounds(
        contours,
        margin=margin,
        image_shape=image.shape,
    )

    cropped = image[y_min:y_max, x_min:x_max]

    if return_offsets:
        return cropped, (y_min, x_min)
    return cropped
