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

    total = np.zeros((ny, nx), dtype=float)

    for contour in contours:
        if len(contour) < 3:
            continue

        # Determine orientation
        area = contour_signed_area(contour)
        sign = np.sign(area)
        if sign == 0:
            continue

        # Create temporary surface per contour
        surface = cairo.ImageSurface(cairo.FORMAT_A8, nx, ny)
        ctx = cairo.Context(surface)

        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.set_source_rgba(1, 1, 1, 1)

        ctx.new_path()
        y0, x0 = contour[0]
        ctx.move_to(x0, y0)
        for y, x in contour[1:]:
            ctx.line_to(x, y)
        ctx.close_path()

        ctx.fill()

        buf = surface.get_data()
        stride = surface.get_stride()

        mask = np.frombuffer(buf, dtype=np.uint8)
        mask = mask.reshape((ny, stride))[:, :nx]
        mask = mask.astype(float) / 255.0

        total += sign * mask

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

    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.set_source_rgba(1, 1, 1, 1)

    for contour in contours:
        if len(contour) == 0:
            continue

        ctx.new_path()

        y0, x0 = contour[0]
        ctx.move_to(x0, y0)
        for y, x in contour[1:]:
            ctx.line_to(x, y)

        ctx.close_path()

    ctx.fill()

    buf = surface.get_data()
    stride = surface.get_stride()

    mask = np.frombuffer(buf, dtype=np.uint8)
    mask = mask.reshape((ny, stride))[:, :nx]

    return mask.astype(dtype) / 255.0
