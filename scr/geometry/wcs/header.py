import numpy as np
import astropy.units as u

from scr.utils.types_alias import Header

from scr.geometry.solar.time import parse_time_to_astropy


def fill_header_for_wcs(
        header: Header
) -> Header:
    """
    Fill in missing keywords and ensure the header is complete for WCS.
    """

    header.setdefault("RSUN_OBS", header.get("SOLAR_RA", 960.))  # In arcsec
    header.comments["RSUN_OBS"] = "[arcsec] angular radius of Sun."
    header.setdefault("DSUN_OBS", 1.0 * u.AU.to(u.m))  # Assume distance to Sun is 1 AU.
    header.comments["DSUN_OBS"] = "[m] Distance from SDO to Sun center."

    header.setdefault("WCSNAME", "Helioprojective-cartesian")
    header.comments["WCSNAME"] = "WCS system name "
    header.setdefault("RSUN_REF", 696000000.)
    header.comments["RSUN_REF"] = "[m] Reference radius of the Sun: 696,000,000.0"
    header.setdefault("DSUN_REF", 1.0 * u.AU.to(u.m))
    header.comments["DSUN_REF"] = "[m] Astronomical Unit"

    header.setdefault("CRLN_OBS", 0.0)  # Default Carrington longitude
    header.comments["CRLN_OBS"] = "[deg] Carrington longitude of the observer"
    header.setdefault("CRLT_OBS", header.get("B_ANGLE", 0.0))  # Use B_ANGLE for CRLT_OBS.
    header.comments["CRLT_OBS"] = "[deg] Carrington latitude of the observer"

    header.setdefault("CROTA2", -header.get("P_ANGLE", -0.))  # Translate P_ANGLE to CROTA2
    header.comments["CROTA2"] = "[deg] CROTA2: INST_ROT + SAT_ROT"

    header.setdefault("CDELT1", header.get("XSCALE", None))  # Pixel scale in arcsec/pixel
    header.comments["CDELT1"] = "[arcsec/pixel] image scale in the x direction"
    header.setdefault("CDELT2", header.get("YSCALE", None))  # Pixel scale in arcsec/pixel
    header.comments["CDELT2"] = "[arcsec/pixel] image scale in the y direction"

    crota2 = np.deg2rad(header["CROTA2"])

    # Set reference point to [0, 0] arcsec by default
    header.setdefault("CRVAL1", 0.)
    header.comments["CRVAL1"] = "[arcsec] CRVAL1: x origin"
    header.setdefault("CRVAL2", 0.)
    header.comments["CRVAL2"] = "[arcsec] CRVAL2: y origin"

    # Convert center and scale information into the WCS-required format
    # http://jsoc.stanford.edu/doc/keywords/JSOC_Keywords_for_metadata.pdf

    crpix1, crpix2 = None, None
    # compute crpix1, crpix2
    if all(k not in header for k in ["CRPIX1", "CRPIX2"]) and all(
            k in header for k in ["NAXIS1", "NAXIS2", "XCEN", "YCEN"]):
        # Solved using determinants
        # y1 = a1 * crpix1 + b1 * crpix2  # XCEN = eq.
        # y2 = a2 * crpix1 + b2 * crpix2  # YCEN = eq.
        y1 = (header["XCEN"] - header["CRVAL1"]
              - header["CDELT1"] * np.cos(crota2) * (header["NAXIS1"] + 1.) / 2.
              + header["CDELT2"] * np.sin(crota2) * (header["NAXIS2"] + 1.) / 2.
              )
        y2 = (header["YCEN"] - header["CRVAL2"]
              - header["CDELT1"] * np.sin(crota2) * (header["NAXIS1"] + 1.) / 2.
              - header["CDELT2"] * np.cos(crota2) * (header["NAXIS2"] + 1.) / 2.
              )

        a1 = -header["CDELT1"] * np.cos(crota2)
        a2 = -header["CDELT1"] * np.sin(crota2)

        b1 = header["CDELT2"] * np.sin(crota2)
        b2 = -header["CDELT2"] * np.cos(crota2)

        det_main = a1 * b2 - a2 * b1
        if det_main != 0.:
            det_crpix1 = y1 * b2 - y2 * b1
            det_crpix2 = a1 * y2 - a2 * y1
            crpix1 = det_crpix1 / det_main
            crpix2 = det_crpix2 / det_main

    xcen, ycen = None, None
    # compute xcen, ycen
    if all(k not in header for k in ["XCEN", "YCEN"]) and all(
            k in header for k in ["NAXIS1", "NAXIS2", "CRPIX1", "CRPIX2"]):
        xcen = (header["CRVAL1"]
                + header["CDELT1"] * np.cos(crota2) * ((header["NAXIS1"] + 1.) / 2. - header["CRPIX1"])
                - header["CDELT2"] * np.sin(crota2) * ((header["NAXIS2"] + 1.) / 2. - header["CRPIX2"]))

        ycen = (header["CRVAL2"]
                + header["CDELT1"] * np.sin(crota2) * ((header["NAXIS1"] + 1.) / 2. - header["CRPIX1"])
                + header["CDELT2"] * np.cos(crota2) * ((header["NAXIS2"] + 1.) / 2. - header["CRPIX2"]))

    header.setdefault("CRPIX1", crpix1)
    header.comments["CRPIX1"] = "[pixel] CRPIX1: location of the Sun center in C"
    header.setdefault("CRPIX2", crpix2)
    header.comments["CRPIX2"] = "[pixel] CRPIX2: location of the Sun center in C"

    header.setdefault("XCEN", xcen)
    header.comments["XCEN"] = "[arcsec] XCEN: location of the Sun center in C"
    header.setdefault("YCEN", ycen)
    header.comments["YCEN"] = "[arcsec] YCEN: location of the Sun center in C"

    # Set default units and types
    header.setdefault("CUNIT1", "arcsec")
    header.comments["CUNIT1"] = "[arcsec] CUNIT1: arcsec"
    header.setdefault("CUNIT2", "arcsec")
    header.comments["CUNIT2"] = "[arcsec] CUNIT2: arcsec"

    header.setdefault("CTYPE1", "HPLN-TAN")
    header.comments["CTYPE1"] = "CTYPE1: HPLN"
    header.setdefault("CTYPE2", "HPLT-TAN")
    header.comments["CTYPE2"] = "CTYPE2: HPLT"

    # Ensure DATE-OBS or TSTART is available as obstime
    header.setdefault("DATE-OBS", header.get("TSTART", None))
    header.comments["DATE-OBS"] = "[ISO] Observation date {DATE__OBS}"

    if header["DATE-OBS"] is not None:
        header["MJD-OBS"] = parse_time_to_astropy(header["DATE-OBS"]).mjd
    else:
        header["MJD-OBS"] = None
    header.comments["MJD-OBS"] = "[d] MJD of fiducial time"

    # Ensure T_OBS is available, using TSTART and TEND
    t_obs = None  # Default if TSTART and TEND aren't present or parsing fails
    if "TSTART" in header and "TEND" in header:
        # Convert to datetime objects
        tstart = parse_time_to_astropy(header["TSTART"])
        tend = parse_time_to_astropy(header["TEND"])

        if tstart and tend:
            # Compute T_OBS as the midpoint
            t_obs = tstart + (tend - tstart) / 2.

            # Convert back to string if needed
            t_obs = t_obs.strftime("%Y-%m-%dT%H:%M:%S.%f")

    header.setdefault("T_OBS", t_obs)
    header.comments["T_OBS"] = "[TAI] nominal time"

    return header
