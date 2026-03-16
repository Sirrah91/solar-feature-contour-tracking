import numpy as np
from skimage.draw import polygon
from typing import Literal

from scr.utils.types_alias import Contour, Contours, Mask
from scr.geometry.raster.fill import contours_to_fractional_mask
from scr.geometry.contours.normalization import normalize_contour_input


def contours_to_binary_mask(
    contours: Contour | Contours,
    shape: tuple[int, int],
    *,
    method: Literal["cairo", "skimage"] = "skimage",
    threshold: float = 0.5,
    dtype: type = bool,
) -> Mask:
    """
    Rasterise contours into a full binary mask.

    Parameters
    ----------
    contours : list of (N, 2) arrays
    shape : (height, width)
    method : {"cairo", "skimage"}
        cairo   -> topology-safe EVEN_ODD fill
        skimage -> simple polygon fill (no hole logic)
    threshold : float
        Used only for cairo method.
    dtype : numpy dtype

    Returns
    -------
    mask : ndarray
    """
    contours = normalize_contour_input(contours)

    if method == "cairo":
        frac = contours_to_fractional_mask(contours, shape)
        mask = frac >= threshold
        return mask.astype(dtype)

    elif method == "skimage":
        mask = np.zeros(shape, dtype=bool)

        for contour in contours:
            if len(contour) == 0:
                continue

            rr, cc = polygon(
                contour[:, 0],
                contour[:, 1],
                shape=shape
            )
            mask[rr, cc] = True

        return mask.astype(dtype)

    else:
        raise ValueError(f"Unknown method: {method}")
