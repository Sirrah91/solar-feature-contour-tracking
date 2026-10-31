"""
Reconstruct WCS + solar-geometry + compatibility keywords in a header
"""
from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, ITRS, SkyCoord, Longitude, Latitude
from sunpy.coordinates import sun, HeliographicCarrington, HeliographicStonyhurst

from src.geometry.solar.time import parse_time_to_astropy
from src.geometry.wcs.compatibility import apply_fits_compatibility
from src.geometry.wcs.utils import safe_round, HeaderReconstructionError
from src.geometry.wcs.constants import RSUN_REF, DSUN_REF
from src.utils.types_alias import Header
from src.utils.logging import get_logger

logger = get_logger("geometry.wcs.header")


def _dig(header: Header, *keys: str):
    for key in keys:
        if key in header and header[key] not in (None, ""):
            return header[key]
    return None


def _require(header: Header, *keys: str, what: str):
    value = _dig(header, *keys)
    if value is None:
        raise HeaderReconstructionError(
            f"Cannot reconstruct {what}: none of {keys!r} present in header."
        )
    return value


# ------------------------------------------------------------------
# obstime
# ------------------------------------------------------------------

def _obstime_from_header(header: Header) -> Time:
    direct = _dig(header, "DATE-OBS", "DATE-BEG", "TSTART")
    if direct is not None:
        return parse_time_to_astropy(direct, default_scale="utc")

    raise HeaderReconstructionError(
        "Cannot reconstruct obstime: none of DATE-OBS/DATE-BEG/TSTART present."
    )


# ------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------

@dataclass(frozen=True)
class _Geometry:
    crpix1: float
    crpix2: float
    crval1: float
    crval2: float
    cdelt1: float
    cdelt2: float
    mean_cdelt: float


def _geometry_from_header(header: Header) -> _Geometry:
    cdelt1 = _dig(header, "CDELT1", "XSCALE")
    if cdelt1 is None:
        raise HeaderReconstructionError("Cannot reconstruct axis-1 pixel scale: neither CDELT1 nor XSCALE present.")
    cdelt2 = _dig(header, "CDELT2", "YSCALE")
    if cdelt2 is None:
        raise HeaderReconstructionError("Cannot reconstruct axis-2 pixel scale: neither CDELT2 nor YSCALE present.")
    cdelt1, cdelt2 = float(cdelt1), float(cdelt2)

    crpix1 = float(_require(header, "CRPIX1", what="CRPIX1"))
    crpix2 = float(_require(header, "CRPIX2", what="CRPIX2"))

    crval1 = float(_require(header, "CRVAL1", what="CRVAL1"))
    crval2 = float(_require(header, "CRVAL2", what="CRVAL2"))

    mean_cdelt = (cdelt1 + cdelt2) / 2.0

    return _Geometry(
        crpix1=crpix1, crpix2=crpix2,
        crval1=crval1, crval2=crval2,
        cdelt1=cdelt1, cdelt2=cdelt2, mean_cdelt=mean_cdelt
    )


# ------------------------------------------------------------------
# Ephemeris -- always computed from obstime
# ------------------------------------------------------------------

@dataclass(frozen=True)
class _Ephemeris:
    obstime: Time
    l0: Longitude
    b0: Latitude
    p_angle: u.Quantity
    radius: u.Quantity  # angular


def _ephemeris_from_time(obstime: Time) -> _Ephemeris:
    return _Ephemeris(
        obstime=obstime,
        l0=Longitude(sun.L0(obstime)),
        b0=Latitude(sun.B0(obstime)),
        p_angle=sun.P(obstime),
        radius=sun.angular_radius(obstime),
    )


def _sanity_check_tier_c(header: Header, ephemeris: _Ephemeris, tolerance_deg: float = 0.2) -> None:
    """
    If the header happens to carry its own B_ANGLE/P_ANGLE/SOLAR_RA, compare
    them against the computed ephemeris and log if they disagree more than
    `tolerance_deg` -- useful as a check that obstime parsed correctly, or
    that this instrument's own values are meaningfully different from the
    generic Earth-centred ones. Never used to override the computed values.
    """
    checks = [
        ("B_ANGLE", ephemeris.b0.to(u.deg).value),
        ("P_ANGLE", ephemeris.p_angle.to(u.deg).value),
    ]
    for key, computed in checks:
        header_value = _dig(header, key)
        if header_value is not None and abs(float(header_value) - computed) > tolerance_deg:
            logger.warning(
                "Tier-C sanity check: header %s=%.4f deg differs from computed "
                "value %.4f deg by more than %.2f deg.",
                key, float(header_value), computed, tolerance_deg,
            )

    rsun_header = _dig(header, "SOLAR_RA", "RSUN_OBS")
    rsun_computed = ephemeris.radius.to(u.arcsec).value
    if rsun_header is not None and abs(float(rsun_header) - rsun_computed) > 5.0:
        logger.warning(
            "Tier-C sanity check: header solar radius=%.2f\" differs from computed "
            "%.2f\" by more than 5\".", float(rsun_header), rsun_computed,
        )


# ------------------------------------------------------------------
# Observer position
# ------------------------------------------------------------------

@dataclass(frozen=True)
class _Observer:
    crln_obs: Longitude
    crlt_obs: Latitude
    hgln_obs: Longitude
    hglt_obs: Latitude
    dsun_obs: u.Quantity  # length
    tier: str  # "direct" | "position" | "earth_center" -- for logging/inspection


def _observer_from_position(header: Header, obstime: Time) -> _Observer | None:
    if all(k in header for k in ("OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z")):
        x, y, z = header["OBSGEO-X"] * u.m, header["OBSGEO-Y"] * u.m, header["OBSGEO-Z"] * u.m
    elif all(k in header for k in ("OBSGEO-L", "OBSGEO-B", "OBSGEO-H")):
        loc = EarthLocation.from_geodetic(
            lon=header["OBSGEO-L"] * u.deg, lat=header["OBSGEO-B"] * u.deg, height=header["OBSGEO-H"] * u.m,
        )
        x, y, z = loc.x, loc.y, loc.z
    else:
        return None

    itrs = SkyCoord(x=x, y=y, z=z, representation_type="cartesian", frame=ITRS(obstime=obstime))
    carrington = itrs.transform_to(HeliographicCarrington(obstime=obstime, observer="sun"))
    stonyhurst = itrs.transform_to(HeliographicStonyhurst(obstime=obstime))

    return _Observer(
        crln_obs=Longitude(carrington.lon),
        crlt_obs=Latitude(carrington.lat),
        hgln_obs=Longitude(stonyhurst.lon),
        hglt_obs=Latitude(stonyhurst.lat),
        dsun_obs=carrington.radius.to(u.m),
        tier="position",
    )


def _observer_from_header(header: Header, obstime: Time, ephemeris: _Ephemeris) -> _Observer:
    crln = _dig(header, "CRLN_OBS")
    crlt = _dig(header, "CRLT_OBS")
    if crln is not None and crlt is not None:
        logger.info("Observer position: using CRLN_OBS/CRLT_OBS directly from header (tier A).")
        crln_obs = Longitude(float(crln) * u.deg)
        crlt_obs = Latitude(float(crlt) * u.deg)
        dsun = _dig(header, "DSUN_OBS")
        dsun_obs = (float(dsun) * u.m) if dsun is not None else sun.earth_distance(obstime).to(u.m)
        return _Observer(
            crln_obs=crln_obs,
            crlt_obs=crlt_obs,
            hgln_obs=Longitude(crln_obs - ephemeris.l0),  # exact, by definition of L0
            hglt_obs=Latitude(crlt_obs),  # exact, latitude is frame-invariant
            dsun_obs=dsun_obs,
            tier="direct",
        )

    from_position = _observer_from_position(header, obstime)
    if from_position is not None:
        logger.info("Observer position: computed exactly from OBSGEO-* in header (tier B).")
        return from_position

    logger.warning(
        "Observer position: no CRLN_OBS/CRLT_OBS and no OBSGEO-* in header -- "
        "assuming observer at Earth's centre (tier C, reusing computed ephemeris)."
    )
    _sanity_check_tier_c(header, ephemeris)
    return _Observer(
        crln_obs=ephemeris.l0,
        crlt_obs=Latitude(ephemeris.b0),
        hgln_obs=Longitude(0.0 * u.deg),
        hglt_obs=Latitude(ephemeris.b0),
        dsun_obs=sun.earth_distance(obstime).to(u.m),
        tier="earth_center",
    )


# ------------------------------------------------------------------
# Header writing
# ------------------------------------------------------------------

def _write_geometry(header: Header, geometry: _Geometry) -> None:
    header.setdefault("WCSAXES", (2, "Number of WCS axes"))
    header.setdefault("WCSNAME", ("Helioprojective-Cartesian", "WCS coordinate system"))
    header.setdefault("CTYPE1", ("HPLN-TAN", "Helioprojective longitude"))
    header.setdefault("CTYPE2", ("HPLT-TAN", "Helioprojective latitude"))
    header.setdefault("CUNIT1", ("arcsec", "Axis 1 unit"))
    header.setdefault("CUNIT2", ("arcsec", "Axis 2 unit"))
    header.setdefault("CRPIX1", (safe_round(geometry.crpix1), "[px] Reference pixel coordinate of axis 1"))
    header.setdefault("CRPIX2", (safe_round(geometry.crpix2), "[px] Reference pixel coordinate of axis 2"))
    header.setdefault("CRVAL1", (safe_round(geometry.crval1), "[arcsec] Coordinate at CRPIX1"))
    header.setdefault("CRVAL2", (safe_round(geometry.crval2), "[arcsec] Coordinate at CRPIX2"))
    header.setdefault("CDELT1", (safe_round(geometry.cdelt1), "[arcsec/px] Image scale of axis 1"))
    header.setdefault("CDELT2", (safe_round(geometry.cdelt2), "[arcsec/px] Image scale of axis 2"))


def _write_rotation(header: Header, p_angle: u.Quantity) -> None:
    existing = _dig(header, "CROTA2")
    angle = (float(existing) * u.deg) if existing is not None else (-p_angle)

    header.setdefault("CROTA2", (safe_round(angle.to(u.deg).value), "[deg] Rotation angle"))

    rad = angle.to(u.rad).value
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    header.setdefault("PC1_1", (safe_round(cos_a), "cos(angle)"))
    header.setdefault("PC1_2", (safe_round(-sin_a), "-sin(angle)"))
    header.setdefault("PC2_1", (safe_round(sin_a), "sin(angle)"))
    header.setdefault("PC2_2", (safe_round(cos_a), "cos(angle)"))


def _write_ephemeris(header: Header, ephemeris: _Ephemeris, mean_cdelt: float) -> None:
    header.setdefault("SOLAR_L0", (safe_round(ephemeris.l0.to(u.deg).value), "[deg] Solar L0 angle"))
    header.setdefault("SOLAR_B0", (safe_round(ephemeris.b0.to(u.deg).value), "[deg] Solar B0 angle"))
    header.setdefault("SOLAR_P0", (safe_round(ephemeris.p_angle.to(u.deg).value), "[deg] Solar P angle"))
    header.setdefault("RSUN_OBS", (safe_round(ephemeris.radius.to(u.arcsec).value), "[arcsec] Observed solar radius"))

    # SOLAR_R (px) belongs here, not to "geometry" -- it's ephemeris.radius
    # converted using the pixel scale. Mean of the two axes since a single
    # px radius is inherently approximate once CDELT1 != CDELT2.
    header.setdefault(
        "SOLAR_R",
        (safe_round(ephemeris.radius.to(u.arcsec).value / mean_cdelt),
         "[px] Observed solar radius (mean of per-axis scale)"),
    )


def _write_observer(header: Header, observer: _Observer) -> None:
    header.setdefault("CRLN_OBS", (safe_round(observer.crln_obs.to(u.deg).value), "[deg] Carrington longitude of observer"))
    header.setdefault("CRLT_OBS", (safe_round(observer.crlt_obs.to(u.deg).value), "[deg] Carrington latitude of observer"))
    header.setdefault("HGLN_OBS", (safe_round(observer.hgln_obs.to(u.deg).value), "[deg] Stonyhurst heliographic longitude"))
    header.setdefault("HGLT_OBS", (safe_round(observer.hglt_obs.to(u.deg).value), "[deg] Stonyhurst heliographic latitude"))
    header.setdefault("DSUN_OBS", (safe_round(observer.dsun_obs.to(u.m).value), "[m] Distance from observer to Sun center"))


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def rebuild_wcs(header: Header) -> Header:
    """
    Reconstruct WCS + solar-geometry + compatibility keywords in `header`,
    in place. Returns the same header object.
    """
    obstime = _obstime_from_header(header)
    ephemeris = _ephemeris_from_time(obstime)
    observer = _observer_from_header(header, obstime, ephemeris)
    geometry = _geometry_from_header(header)

    _write_geometry(header, geometry)
    _write_rotation(header, ephemeris.p_angle)
    _write_ephemeris(header, ephemeris, geometry.mean_cdelt)
    _write_observer(header, observer)

    header.setdefault("RSUN_REF", (safe_round(RSUN_REF.m), "[m] Reference solar radius"))
    header.setdefault("DSUN_REF", (safe_round(DSUN_REF.m), "[m] Reference Sun-Earth distance"))

    apply_fits_compatibility(header)

    return header


def fill_header_for_wcs(
        header: Header,
        *,
        recompute_legacy_keywords: bool = False,
) -> Header:
    """
    Prepare a FITS header for WCS creation.

    By default, delegates directly to the robust WCS reconstruction.
    Legacy keyword recomputation is available only as an explicit opt-in
    for exceptional input data.
    """
    if recompute_legacy_keywords:
        from src.geometry.wcs.compatibility import recompute_legacy_keywords
        logger.warning(
            "Recomputing legacy WCS-related keywords before WCS reconstruction. "
            "This is an exceptional compatibility mode and is not recommended "
            "for normal use."
        )
        header = recompute_legacy_keywords(header)

    return rebuild_wcs(header)
