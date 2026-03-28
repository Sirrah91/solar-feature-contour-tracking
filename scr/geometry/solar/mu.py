import numpy as np
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Helioprojective, HeliographicStonyhurst

from scr.utils.types_alias import Header

from scr.geometry.solar.time import parse_time_to_astropy


def compute_mu(
        header: Header
) -> np.ndarray:
    return compute_mu_radial(header)


def compute_mu_radial(
        header: Header
) -> np.ndarray:
    """
    Fast radial approximation of mu assuming a spherical solar disk.
    """
    nx, ny = header["NAXIS1"], header["NAXIS2"]

    x = np.arange(nx) - (header["CRPIX1"] - 1.0)
    y = np.arange(ny) - (header["CRPIX2"] - 1.0)
    xx, yy = np.meshgrid(x, y)

    x_arcsec = xx * header["CDELT1"]
    y_arcsec = yy * header["CDELT2"]

    r2 = (x_arcsec**2 + y_arcsec**2) / header["RSUN_OBS"]**2

    mu = np.full_like(r2, np.nan, dtype=float)
    inside = r2 <= 1.0
    mu[inside] = np.sqrt(1.0 - r2[inside])

    return mu


def compute_mu_observer(
        header: Header
) -> np.ndarray:
    """
    Compute mu using full observer geometry and WCS.
    """
    wcs = WCS(header)

    dsun_obs = header["DSUN_OBS"] * u.m
    crln_obs = header["CRLN_OBS"] * u.deg
    crlt_obs = header["CRLT_OBS"] * u.deg
    obstime = parse_time_to_astropy(header["DATE-OBS"])

    observer = SkyCoord(
        lon=crln_obs,
        lat=crlt_obs,
        radius=dsun_obs,
        frame=HeliographicStonyhurst,
        obstime=obstime,
    )

    nx, ny = header["NAXIS1"], header["NAXIS2"]
    y, x = np.indices((ny, nx))

    hpc = wcs.pixel_to_world(x, y)
    hpc = SkyCoord(
        hpc.Tx, hpc.Ty,
        frame=Helioprojective(observer=observer, obstime=obstime)
    )

    hgs = hpc.transform_to(HeliographicStonyhurst(obstime=obstime))

    # cos(theta) between local vertical and line of sight
    mu = np.cos(hgs.separation(observer).to(u.rad))

    r_arcsec = np.asarray(np.hypot(hpc.Tx, hpc.Ty))
    mu[r_arcsec > header["RSUN_OBS"]] = np.nan

    return np.asarray(mu)
