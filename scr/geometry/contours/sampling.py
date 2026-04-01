import numpy as np
from scipy.ndimage import map_coordinates

from scr.utils.types_alias import Contour

from scr.geometry.contours.length import compute_contour_arc_lengths


def sample_map_at_contour(
        contour: Contour,
        data_map: np.ndarray,
        interp: bool = True
) -> np.ndarray:
    """
    Sample 2D data_map at ordered contour coordinates.
    contour: (N,2) array of (row, col) coordinates (can be floats for subpixel).
    data_map: 2D array
    interp: if True use bilinear interpolation (map_coordinates). If False use nearest (int indices).
    Returns: 1D array length N
    """

    if interp:
        # map_coordinates expects (row_coords, col_coords)
        coords = np.vstack([contour[:, 0], contour[:, 1]])
        sampled = map_coordinates(data_map, coords, order=1, mode="nearest")
        return sampled
    else:
        # nearest (round) integer indexing, preserves order
        r = np.round(contour[:, 0]).astype(int)
        c = np.round(contour[:, 1]).astype(int)
        return data_map[r, c]


def calc_arc_lengths(
        contour: Contour,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        rsun: float
) -> np.ndarray:
    lon = sample_map_at_contour(contour=contour, data_map=lon2d, interp=True)
    lat = sample_map_at_contour(contour=contour, data_map=lat2d, interp=True)

    ds = compute_contour_arc_lengths(lon_deg=lon, lat_deg=lat, rsun=rsun)
    ds = 0.5 * (ds + np.roll(ds, 1))  # centre of the arc

    return ds


def calc_spherical_length(
        contour_lon: np.ndarray,
        contour_lat: np.ndarray,
        rsun: float
) -> float:
    ds = compute_contour_arc_lengths(lon_deg=contour_lon, lat_deg=contour_lat, rsun=rsun)

    return float(np.nansum(ds))
