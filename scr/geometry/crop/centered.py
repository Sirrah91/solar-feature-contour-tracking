import numpy as np

from scr.utils.types_alias import Contour, Contours
from scr.utils.filesystem import is_empty

from scr.geometry.align import shift_to_centre_and_pad
from scr.geometry.contours.normalization import normalize_contour_input


def crop_centered_fixed(
        image: np.ndarray,
        contours: Contour | Contours,
        *,
        target_shape: tuple[int, int],
        background_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop a fixed-size window centred on contour centroid.
    """
    if is_empty(contours):
        raise ValueError("contours must contain at least one contour")

    contours = normalize_contour_input(contours)
    points = np.vstack(contours)
    cy, cx = np.mean(points, axis=0)

    cropped, new_cy, new_cx = shift_to_centre_and_pad(
        image,
        cy, cx,
        target_shape,
        background_value=background_value,
        return_centre=True,
    )

    shift = np.array([cy, cx]) - np.array([new_cy, new_cx])
    return cropped, shift
