from typing import Literal
import numpy as np

from scr.config.numerics import WP
from scr.utils.types_alias import Contour, Contours, Mask
from scr.geometry.contours.normalization import normalize_contour_input
from scr.geometry.contours.shapes import union_contours, shape_to_contours

from scr.geometry.raster.engines import cairo as cairo_engine
from scr.geometry.raster.engines import skimage as skimage_engine

RasterMode = Literal["surface", "border"]
Engine = Literal["cairo", "skimage"]


def rasterize(
        contours: Contour | Contours,
        shape: tuple[int, int],
        *,
        mode: RasterMode = "surface",
        engine: Engine = "cairo",
        use_orientation: bool = True,
        threshold: float = 0.5,
        dtype: type = WP,
        **kwargs,
) -> Mask:
    """
    Rasterise contours into a mask.

    Parameters
    ----------
    contours : Contour or sequence of Contour
        Input contours as arrays of shape (N, 2) with (row, col) coordinates.
    shape : tuple of int
        Output mask shape (ny, nx).
    mode : {"surface", "border"}, optional
        "surface" returns filled regions.
        "border" returns contour boundaries.
    engine : {"cairo", "skimage"}, optional
        Rasterisation backend:
        - "cairo": anti-aliased, subpixel precision
        - "skimage": discrete pixel rasterisation
    use_orientation : bool, optional
        If True, contour orientation defines holes (CW subtracts).
        If False, all contours are treated as filled (holes removed via union).
    threshold : float, optional
        Threshold used when dtype is bool.
    dtype : type, optional
        Output dtype. If bool, thresholding is applied.

    Returns
    -------
    mask : ndarray
        Rasterised mask of shape `shape`.
    """
    contours = normalize_contour_input(contours)

    if not use_orientation:  # Remove interior structure from contours
        poly = union_contours(contours)
        if poly is None:
            return np.zeros(shape, dtype=dtype)

        contours = shape_to_contours(poly, enforce_ccw=True)
        contours = normalize_contour_input(contours)

    if engine == "cairo":
        mask = _rasterize_cairo(
            contours, shape, mode, **kwargs
        )
    elif engine == "skimage":
        mask = _rasterize_skimage(
            contours, shape, mode, **kwargs
        )
    else:
        raise ValueError(f"Unknown engine: {engine}")

    if dtype == bool:  # return boolean mask
        return mask >= threshold

    return mask.astype(dtype)


# -------------------------
# Engine dispatchers
# -------------------------

def _rasterize_cairo(
        contours: Contours,
        shape: tuple[int, int],
        mode: RasterMode,
        **kwargs
) -> Mask:
    if mode == "surface":
        return cairo_engine.surface(contours, shape, dtype=WP)

    elif mode == "border":
        return cairo_engine.border(contours, shape, dtype=WP, **kwargs)

    else:
        raise ValueError(
            f"Mode '{mode}' not supported by cairo engine"
        )


def _rasterize_skimage(
        contours: Contours,
        shape: tuple[int, int],
        mode: RasterMode,
        **kwargs
) -> Mask:
    if mode == "surface":
        return skimage_engine.surface(contours, shape, dtype=WP)

    elif mode == "border":
        return skimage_engine.border(contours, shape, dtype=WP, **kwargs)

    else:
        raise ValueError(
            f"Mode '{mode}' not supported by skimage engine"
        )
