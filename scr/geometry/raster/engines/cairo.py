import numpy as np
import cairo

from scr.config.numerics import WP
from scr.utils.types_alias import Contour, Contours, Mask
from scr.geometry.contours.orientation import is_ccw


# -------------------------
# Shared utilities
# -------------------------

def _create_surface(
        shape: tuple[int, int],
) -> tuple[cairo.ImageSurface, cairo.Context]:
    """
    Create a Cairo A8 surface and context.

    Parameters
    ----------
    shape : tuple of int
        Output shape (ny, nx).

    Returns
    -------
    surface : cairo.ImageSurface
    ctx : cairo.Context
    """
    ny, nx = shape
    surf = cairo.ImageSurface(cairo.FORMAT_A8, nx, ny)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    return surf, ctx


def _get_buffer_view(
        surf: cairo.ImageSurface,
        shape: tuple[int, int],
) -> np.ndarray:
    """
    Return a uint8 view of the Cairo surface buffer.

    Parameters
    ----------
    surf : cairo.ImageSurface
    shape : tuple of int

    Returns
    -------
    buf : ndarray
        View of shape (ny, nx), values in [0, 255].
    """
    ny, nx = shape
    stride = surf.get_stride()

    buf = surf.get_data()
    arr = np.frombuffer(buf, dtype=np.uint8)

    return arr.reshape((ny, stride))[:, :nx]


def _draw_path(
        ctx: cairo.Context,
        contour: Contour,
) -> None:
    """
    Add a closed contour path to the Cairo context.
    """
    # Added the 0.5px offset for center-to-corner alignment
    y0, x0 = contour[0]
    ctx.move_to(x0 + 0.5, y0 + 0.5)

    for y, x in contour[1:]:
        ctx.line_to(x + 0.5, y + 0.5)

    ctx.close_path()


# -------------------------
# Public API
# -------------------------

def surface(
        contours: Contour | Contours,
        shape: tuple[int, int],
        *,
        dtype: type = WP,
) -> Mask:
    """
    Rasterise contours into a fractional surface mask.

    Parameters
    ----------
    contours : sequence of Contour
    shape : tuple of int
    dtype : type, optional

    Returns
    -------
    mask : ndarray
        Float mask in [0, 1].
    """
    surf, ctx = _create_surface(shape)

    # Get reusable buffer view
    buf = _get_buffer_view(surf, shape)

    total = np.zeros(shape, dtype=float)

    # Set once
    ctx.set_source_rgba(1, 1, 1, 1)

    for contour in contours:
        if len(contour) < 3:
            continue

        # CCW adds, CW subtracts
        sign = 1.0 if is_ccw(contour) else -1.0

        # FAST CLEAR (no cairo operator)
        buf[:] = 0
        surf.mark_dirty()

        ctx.new_path()
        _draw_path(ctx, contour)
        ctx.fill()

        # Ensure Cairo finished writing
        surf.flush()

        # No reallocation, reuse buffer
        total += sign * buf.astype(float)

    return np.clip(total / 255.0, 0.0, 1.0).astype(dtype)


def border(
        contours: Contours,
        shape: tuple[int, int],
        *,
        dtype: type = WP,
        line_width: float = 1.0,
) -> Mask:
    """
    Rasterise contour borders.

    Parameters
    ----------
    contours : sequence of Contour
    shape : tuple of int
    line_width : float, optional

    Returns
    -------
    mask : ndarray
        Float mask in [0, 1].
    """
    surf, ctx = _create_surface(shape)

    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.set_line_width(line_width)

    for contour in contours:
        if len(contour) < 2:
            continue

        ctx.new_path()
        _draw_path(ctx, contour)
        ctx.stroke()

    # ensure drawing is finished
    surf.flush()

    buf = _get_buffer_view(surf, shape)

    return np.clip(buf.astype(float) / 255.0, 0.0, 1.0).astype(dtype)
