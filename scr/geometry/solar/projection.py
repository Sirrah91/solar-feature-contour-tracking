import numpy as np
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Helioprojective, HeliographicStonyhurst

from scr.utils.types_alias import Header

from scr.geometry.solar.time import parse_time_to_astropy
from scr.geometry.wcs.header import fill_header_for_wcs


def pixel_to_lonlat(
        header: Header
) -> tuple[np.ndarray, np.ndarray]:
    # Fill the header with necessary keywords for WCS
    header = fill_header_for_wcs(header)

    # Create WCS object directly from the complete header
    wcs = WCS(header)

    # Get observer location information from the header
    dsun_obs = header["DSUN_OBS"] * u.m
    crln_obs = header["CRLN_OBS"] * u.deg
    crlt_obs = header["CRLT_OBS"] * u.deg
    obstime = parse_time_to_astropy(header["T_OBS"])

    # Define observer's location in Heliographic Stonyhurst frame
    observer = SkyCoord(
        lon=crln_obs,
        lat=crlt_obs,
        radius=dsun_obs,
        frame=HeliographicStonyhurst,
        obstime=obstime
    )

    # Assuming nx and ny are the dimensions of your image:
    nx, ny = header["NAXIS1"], header["NAXIS2"]
    y, x = np.indices((ny, nx))

    # Convert pixel coordinates to world coordinates (Helioprojective)
    hpc_coords = wcs.pixel_to_world(x, y)

    # Attach observer information to helioprojective coordinates
    hpc_coords = SkyCoord(
        hpc_coords.Tx, hpc_coords.Ty,
        frame=Helioprojective(observer=observer, obstime=obstime)
    )

    # Transform to Heliographic Stonyhurst coordinates
    heliographic_coords = hpc_coords.transform_to(HeliographicStonyhurst(obstime=obstime))

    # Extract longitude and latitude
    lon = heliographic_coords.lon.to(u.deg).value
    lat = heliographic_coords.lat.to(u.deg).value

    # Apply the correction for the Carrington longitude; zero longitude in the central meridian
    lon = (lon + 360. - crln_obs.to(u.deg).value) % 360.
    lon[lon > 180.] -= 360.  # have if from -180 to +180

    lon = np.clip(lon, a_min=-180., a_max=180.)
    lat = np.clip(lat, a_min=-90., a_max=90.)

    return lon, lat
