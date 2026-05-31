import numpy as np
from astropy.wcs import WCS
# must be imported otherwise 'wcs.pixel_to_world' does not recognise the frame as Helioprojective
from sunpy.coordinates import Helioprojective


def axis_ticks_from_resolution(
        n_pixels: int,
        resolution: float,
        step: float | int,
        *,
        centre_phys: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pixel tick positions and physical tick labels for an image axis.

    Parameters
    ----------
    n_pixels : int
        Number of pixels along the axis.
    resolution : float
        Physical size per pixel (e.g. arcsec / pixel).
    step : float
        Tick spacing in physical units.
    centre_phys : float, optional
        Physical coordinate of the central pixel.

    Returns
    -------
    ticks : ndarray
        Tick positions in pixel coordinates.
    labels : ndarray[str]
        Tick labels in physical units.
    """
    # default centre = middle pixel
    centre_pix = (n_pixels - 1) / 2

    # min/max in physical units
    phys_min = (0 - centre_pix) * resolution + centre_phys
    phys_max = ((n_pixels - 1) - centre_pix) * resolution + centre_phys

    # pick multiples of step
    labels = np.arange(
        np.ceil(phys_min / step) * step,
        np.floor(phys_max / step) * step + step,
        step,
        dtype=type(step)
    )

    # convert physical units → pixel index
    ticks = (labels - centre_phys) / resolution + centre_pix

    return ticks, labels.astype(str)


def axis_ticks_from_wcs(
        wcs: WCS,
        shape: tuple[int, int],
        resolution: tuple[float, float],
        step: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Generate x/y ticks and labels (arcsec) for a cropped image using WCS.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        WCS of the cropped image.
    shape : (ny, nx)
        Image shape.
    resolution : (dx, dy)
        Pixel resolution in physical units (arcsec/pixel).
    step : float
        Tick spacing in physical units.

    Returns
    -------
    (x_ticks, x_labels), (y_ticks, y_labels)
    """
    ny, nx = shape
    dx, dy = resolution

    xc = (nx - 1) / 2
    yc = (ny - 1) / 2

    world = wcs.pixel_to_world(xc, yc)

    x_center = world.Tx.value
    y_center = world.Ty.value

    x_ticks, x_labels = axis_ticks_from_resolution(
        nx, dx, step, centre_phys=x_center
    )
    y_ticks, y_labels = axis_ticks_from_resolution(
        ny, dy, step, centre_phys=y_center
    )

    return (x_ticks, x_labels), (y_ticks, y_labels)
