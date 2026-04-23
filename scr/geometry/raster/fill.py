import numpy as np
import cairo

from scr.config.numerics import WP
from scr.utils.types_alias import Contour, Contours, Mask
from scr.geometry.contours.normalization import normalize_contour_input
from scr.geometry.contours.area import contour_signed_area


def contours_to_signed_fractional_mask(
        contours: Contour | Contours,
        shape: tuple[int, int],
        dtype: type = WP,
) -> Mask:
    """
    Rasterise contours into a signed fractional mask.

    CCW contours contribute +1.
    CW contours contribute -1.

    Resulting mask may contain negative values.
    """
    contours = normalize_contour_input(contours)
    ny, nx = shape

    # Create surface and context ONCE
    surface = cairo.ImageSurface(cairo.FORMAT_A8, nx, ny)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    stride = surface.get_stride()

    total = np.zeros((ny, nx), dtype=float)

    for contour in contours:
        if len(contour) < 3:
            continue

        area = contour_signed_area(contour)
        sign = np.sign(area)
        if sign == 0:
            continue

        # Reset the surface for the next contour
        ctx.set_operator(cairo.OPERATOR_CLEAR)
        ctx.paint()

        # Draw the contour
        ctx.set_operator(cairo.OPERATOR_OVER)  # Default drawing mode
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.set_source_rgba(1, 1, 1, 1)

        ctx.new_path()
        # Note: Added the 0.5px offset for center-to-corner alignment
        y0, x0 = contour[0]
        ctx.move_to(x0 + 0.5, y0 + 0.5)
        for y, x in contour[1:]:
            ctx.line_to(x + 0.5, y + 0.5)
        ctx.close_path()
        ctx.fill()

        # Ensure Cairo has finished drawing before NumPy looks at the memory
        surface.flush()

        # Extract data (this is just a view, no copy yet)
        buf = surface.get_data()
        mask = np.frombuffer(buf, dtype=np.uint8)
        mask = mask.reshape((ny, stride))[:, :nx]

        # Accumulate with sign
        total += (sign * mask) / 255.0

    return total.astype(dtype)


def contours_to_fractional_mask(
        contours: Contour | Contours,
        shape: tuple[int, int],
        dtype: type = WP,
) -> Mask:
    """
    Rasterise contours into a fractional filling-factor mask.
    """
    contours = normalize_contour_input(contours)
    ny, nx = shape

    surface = cairo.ImageSurface(cairo.FORMAT_A8, nx, ny)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)

    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.set_source_rgba(1, 1, 1, 1)

    for contour in contours:
        if len(contour) < 3:
            continue

        ctx.new_path()

        # Note: Added the 0.5px offset for center-to-corner alignment
        y0, x0 = contour[0]
        ctx.move_to(x0 + 0.5, y0 + 0.5)
        for y, x in contour[1:]:
            ctx.line_to(x + 0.5, y + 0.5)

        ctx.close_path()

    ctx.fill()
    surface.flush()

    buf = surface.get_data()
    stride = surface.get_stride()

    mask = np.frombuffer(buf, dtype=np.uint8)
    mask = mask.reshape((ny, stride))[:, :nx]

    return (mask.astype(float) / 255.0).astype(dtype)
