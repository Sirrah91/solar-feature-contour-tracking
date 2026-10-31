import numpy as np


def pad_to_center(
        image: np.ndarray,
        cy: float,
        cx: float,
        background_value: float = np.nan,
        return_centre: bool = False
) -> np.ndarray | tuple[np.ndarray, float, float]:
    """
    Pad the image so that (cy, cx) becomes the geometric centre of the output.

    This does not shift pixel values but adds symmetric padding to centre the object
    based on (cy, cx).

    Parameters
    ----------
    image : np.ndarray
        Input 2D array.
    cy, cx : float
        Coordinates of the current object centre.
    background_value : float, default NaN.
        Value used for padding.
    return_centre : bool, default=False
        Whether to return the new object centre after padding.

    Returns
    -------
    padded_image : np.ndarray
        Padded image.
    new_cy, new_cx : float
        New coordinates of the object centre (if `return_centre` is True).
    """
    h, w = image.shape

    # Distance from the current centre to each edge
    top_dist, left_dist = int(np.round(cy)), int(np.round(cx))
    bottom_dist, right_dist = h - top_dist - 1, w - left_dist - 1

    # Compute padding required to equalise distances
    pad_top = max(bottom_dist - top_dist, 0)
    pad_bottom = max(top_dist - bottom_dist, 0)
    pad_left = max(right_dist - left_dist, 0)
    pad_right = max(left_dist - right_dist, 0)

    # Pad image with background values
    padded = np.pad(
        image,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=background_value
    )

    return (padded, cy + pad_top, cx + pad_left) if return_centre else padded


def pad_or_crop(
        image: np.ndarray,
        target_shape: tuple[int, int] | int,
        background_value: float = np.nan,
        return_shifts: bool = False
) -> np.ndarray | tuple[np.ndarray, int, int]:
    """
    Resize an array to a target shape by symmetric padding or cropping.

    This function symmetrically pads the array if the target shape is larger,
    or crops it centrally if the target shape is smaller. Optionally, it can also return
    the number of pixels added (positive) or removed (negative) from each side.

    Parameters
    ----------
    image : np.ndarray
        2D input array to be resized.
    target_shape : tuple[int, int] or int
        Desired output shape. If an integer is provided, a square shape is assumed.
    background_value : float, default NaN.
        The value to set the padding.
    return_shifts : bool, default=False
        If True, return the (y, x) shifts caused by padding or cropping.

    Returns
    -------
    result : np.ndarray
        The resized array.
    shift_y : int, optional
        Number of rows added or removed (if `return_shifts` is True).
    shift_x : int, optional
        Number of columns added or removed (if `return_shifts` is True).
    """
    if isinstance(target_shape, int):
        target_shape = (target_shape, target_shape)

    original_shape = image.shape

    # Resize along the first axis (rows)
    if target_shape[0] > original_shape[0]:
        # Padding: compute symmetric padding and adjust for odd differences
        pad_x1 = (target_shape[0] - original_shape[0]) // 2
        pad_x2 = pad_x1
        pad_x1 += (target_shape[0] - original_shape[0]) % 2  # extra row on top if needed
        array = np.pad(image, pad_width=((pad_x1, pad_x2), (0, 0)), mode="constant", constant_values=background_value)
        shift_y = pad_x1
    else:
        # Cropping: extract central part
        crop_x = (original_shape[0] - target_shape[0]) // 2
        array = image[crop_x:crop_x + target_shape[0], :]
        shift_y = -crop_x

    # Resize along the second axis (columns)
    if target_shape[1] > original_shape[1]:
        # Padding: compute symmetric padding and adjust for odd differences
        pad_y1 = (target_shape[1] - original_shape[1]) // 2
        pad_y2 = pad_y1
        pad_y1 += (target_shape[1] - original_shape[1]) % 2  # extra column on left if needed
        shift_x = pad_y1
        padded = np.pad(array, pad_width=((0, 0), (pad_y1, pad_y2)), mode="constant", constant_values=background_value)
        if return_shifts:
            return padded, shift_y, shift_x
        return padded
    else:
        # Cropping: extract central part
        crop_y = (original_shape[1] - target_shape[1]) // 2
        shift_x = -crop_y
        cropped = array[:, crop_y:crop_y + target_shape[1]]
        if return_shifts:
            return cropped, shift_y, shift_x
        return cropped


def shift_to_centre_and_pad(
        image: np.ndarray,
        cy: float,
        cx: float,
        target_shape: tuple[int, int] | int,
        background_value: float = np.nan,
        return_centre: bool = False
) -> np.ndarray | tuple[np.ndarray, float, float]:
    """
    Shift the image so that (cy, cx) becomes the new centre and resize it to a target shape.

    The image is first padded with NaNs such that the specified object centre (cy, cx)
    becomes the geometric centre. It is then resized to the desired target shape
    by symmetric zero-padding or cropping.

    Parameters
    ----------
    image : np.ndarray
        2D input image.
    cy, cx : float
        Coordinates of the current object centre.
    background_value : float or None, default NaN.
        The value to set the padding.
    target_shape : tuple[int, int] or int
        Desired output shape. If an integer is provided, a square shape is assumed.
    return_centre : bool, default=False
        If True, return the new coordinates of the object centre in the resized image.

    Returns
    -------
    final_image : np.ndarray
        Image with the object centred and resized to target shape.
    final_cy, final_cx : float, optional
        New coordinates of the object centre in the final image.
        Only returned if `return_centre` is True.
    """
    # Shift the image so that (cy, cx) becomes its geometric centre
    centered_image, new_cy, new_cx = pad_to_center(
        image=image,
        cy=cy,
        cx=cx,
        background_value=background_value,
        return_centre=True
    )

    # Resize to target shape using padding or cropping
    final_image, shift_y, shift_x = pad_or_crop(
        image=centered_image,
        target_shape=target_shape,
        return_shifts=True,
        background_value=background_value
    )

    if return_centre:
        # Compute the new position of the object centre after resizing
        return final_image, new_cy + shift_y, new_cx + shift_x

    return final_image
