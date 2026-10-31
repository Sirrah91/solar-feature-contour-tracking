import numpy as np
from astropy.wcs import WCS
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Helioprojective, HeliographicStonyhurst

from src.utils.types_alias import Header

from src.geometry.solar.time import parse_time_to_astropy
from src.geometry.wcs.header import rebuild_wcs


def get_wcs_and_observer(header: Header) -> tuple[WCS, SkyCoord, Time]:
    """One-time per-header setup, meant to be built once per frame and
    reused across every pixel_to_lonlat / lonlat_to_pixel call for it."""
    header = rebuild_wcs(header)
    wcs = WCS(header)
    obstime = parse_time_to_astropy(header["T_OBS"])
    observer = wcs.pixel_to_world(0, 0).observer
    return wcs, observer, obstime


def pixel_to_lonlat(
        header: Header,
        wcs: WCS | None = None,
        obstime: Time | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if wcs is None or obstime is None:
        wcs, _, obstime = get_wcs_and_observer(header)

    nx, ny = header["NAXIS1"], header["NAXIS2"]
    y, x = np.indices((ny, nx))
    hpc_coords = wcs.pixel_to_world(x, y)

    heliographic_coords = hpc_coords.transform_to(HeliographicStonyhurst(obstime=obstime))
    lon = np.clip(heliographic_coords.lon.to(u.deg).value, -180., 180.)
    lat = np.clip(heliographic_coords.lat.to(u.deg).value, -90., 90.)

    return lon, lat


def lonlat_to_pixel(
        lon_deg: np.ndarray | float,
        lat_deg: np.ndarray | float,
        header: Header,
        wcs: WCS | None = None,
        observer: SkyCoord | None = None,
        obstime: Time | None = None,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    Inverse of pixel_to_lonlat. Accepts a scalar or an ndarray of any
    shape for lon_deg/lat_deg (must match each other); result[0]/result[1]
    (row, col) has that same shape, elementwise: (row[i,j], col[i,j])
    comes from (lon_deg[i,j], lat_deg[i,j]). Scalar input still returns
    plain floats, not 0-d arrays, so existing single-point call sites
    (e.g. the plotting code) don't need to change.
    """
    if wcs is None or observer is None or obstime is None:
        wcs, observer, obstime = get_wcs_and_observer(header)

    lon_deg = np.asarray(lon_deg, dtype=float)
    lat_deg = np.asarray(lat_deg, dtype=float)
    if lon_deg.shape != lat_deg.shape:
        raise ValueError(f"lonlat_to_pixel: lon_deg.shape={lon_deg.shape} != lat_deg.shape={lat_deg.shape}")
    scalar_input = lon_deg.ndim == 0

    rsun = header["RSUN_REF"] * u.m
    heliographic_coord = SkyCoord(
        lon=lon_deg * u.deg, lat=lat_deg * u.deg, radius=rsun,
        frame=HeliographicStonyhurst, obstime=obstime,
    )
    hpc_coord = heliographic_coord.transform_to(Helioprojective(observer=observer, obstime=obstime))

    col, row = wcs.world_to_pixel(hpc_coord)
    row, col = np.asarray(row), np.asarray(col)

    if scalar_input:
        return float(row), float(col)
    return row, col


def lonlat_to_xyz(
        lon_deg: np.ndarray,
        lat_deg: np.ndarray,
        rsun: float = 1.0,
) -> np.ndarray:
    """Heliographic lon/lat -> xyz on a sphere of radius rsun."""
    lon_r, lat_r = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    x = rsun * np.cos(lat_r) * np.cos(lon_r)
    y = rsun * np.cos(lat_r) * np.sin(lon_r)
    z = rsun * np.sin(lat_r)
    return np.column_stack([x, y, z])


def local_tangent_frame(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (e1, e2) basis tangent to the sphere at xyz's mean
    position -- flattens a small patch to 2D for area/angle purposes.
    Valid because a spot's angular extent is tiny compared to the sphere."""
    normal = xyz.mean(axis=0)
    normal = normal / np.linalg.norm(normal)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return normal, e1, e2


def flatten(
        xyz: np.ndarray,
        origin: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
) -> np.ndarray:
    d = xyz - origin
    return np.column_stack([d @ e1, d @ e2])
