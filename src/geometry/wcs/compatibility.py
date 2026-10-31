"""
Backward compatibility layer for legacy FITS keywords.

This module must NEVER contain physics or WCS logic.
Only aliasing, duplication, and historical keyword support.
"""
import numpy as np

from src.utils.logging import get_logger
from src.geometry.wcs.utils import alias_keyword
from src.utils.types_alias import Header

logger = get_logger("geometry.wcs.compatibility")


def apply_fits_compatibility(
        header: Header,
) -> Header:
    """
    Adds legacy / external-software FITS keywords.
    """

    # ------------------------------------------------------------
    # Aliases (canonical → legacy)
    # ------------------------------------------------------------

    alias_keyword(header, source="CROTA2", target="ANGLE",
                  comment="[deg] Rotation angle (= CROTA2)")

    alias_keyword(header, source="RSUN_OBS", target="RSUN_ARC",
                  comment="[arcsec] Observed solar radius (= RSUN_OBS)")

    alias_keyword(header, source="CDELT1", target="XSCALE",
                  comment="[arcsec/px] Image scale of axis 1 (= CDELT1)")

    alias_keyword(header, source="CDELT2", target="YSCALE",
                  comment="[arcsec/px] Image scale of axis 2 (= CDELT2)")

    return header


def recompute_legacy_keywords(header: Header) -> Header:
    """
    Convert legacy WCS metadata into canonical WCS keywords.

    This is an exceptional compatibility path for old headers and should
    normally not be used. The regular ``rebuild_wcs()`` path should be
    preferred whenever canonical WCS metadata are already available.

    Legacy input system
    -------------------
    NAXIS1, NAXIS2
    XCEN, YCEN
    XSCALE, YSCALE
    P_ANGLE

    Canonical keywords reconstructed when missing
    ---------------------------------------------
    CRVAL1, CRVAL2
    CDELT1, CDELT2
    CROTA2
    CRPIX1, CRPIX2

    Existing canonical keywords are preserved.

    No geometric quantity is guessed. If the required legacy metadata are
    incomplete, the corresponding reconstruction is skipped.
    """
    # ------------------------------------------------------------------
    # Required legacy metadata.
    #
    # XCEN/YCEN are defined in terms of the legacy pixel scales and
    # CROTA2. Since this is specifically a legacy-system conversion,
    # XSCALE/YSCALE and P_ANGLE are the authoritative inputs here.
    # ------------------------------------------------------------------
    required = (
        "NAXIS1",
        "NAXIS2",
        "XCEN",
        "YCEN",
        "XSCALE",
        "YSCALE",
        "P_ANGLE",
    )

    if not all(key in header for key in required):
        logger.warning(
            "Legacy WCS compatibility: insufficient legacy metadata "
            "to reconstruct canonical WCS keywords."
        )
        return header

    # ------------------------------------------------------------------
    # Convert the legacy pixel scale to canonical CDELT keywords.
    #
    # Do this explicitly here so that rebuild_wcs() does not need to know
    # about the legacy XSCALE/YSCALE aliases.
    # ------------------------------------------------------------------
    if "CDELT1" not in header:
        header["CDELT1"] = (
            float(header["XSCALE"]),
            "[arcsec/px] Reconstructed from XSCALE",
        )

    if "CDELT2" not in header:
        header["CDELT2"] = (
            float(header["YSCALE"]),
            "[arcsec/px] Reconstructed from YSCALE",
        )

    # ------------------------------------------------------------------
    # The legacy XCEN/YCEN convention is relative to the solar disk
    # centre, so use (0, 0) as the canonical helioprojective reference
    # coordinate when CRVAL1/CRVAL2 are absent.
    #
    # Existing values are preserved.
    # ------------------------------------------------------------------
    if "CRVAL1" not in header:
        header["CRVAL1"] = (
            0.0,
            "[arcsec] Reference helioprojective longitude",
        )

    if "CRVAL2" not in header:
        header["CRVAL2"] = (
            0.0,
            "[arcsec] Reference helioprojective latitude",
        )

    # ------------------------------------------------------------------
    # Convert the legacy position angle to CROTA2.
    #
    # The legacy XCEN/YCEN definitions explicitly use CROTA2, while the
    # old headers provide P_ANGLE. The historical conversion is:
    #
    #     CROTA2 = -P_ANGLE
    #
    # Existing CROTA2 is preserved.
    # ------------------------------------------------------------------
    if "CROTA2" not in header:
        header["CROTA2"] = (
            -float(header["P_ANGLE"]),
            "[deg] Reconstructed from -P_ANGLE",
        )

    # ------------------------------------------------------------------
    # CRPIX1/CRPIX2 are already available.
    # ------------------------------------------------------------------
    if "CRPIX1" in header and "CRPIX2" in header:
        return header

    # Do not reconstruct only one member of the CRPIX pair.
    if "CRPIX1" in header or "CRPIX2" in header:
        logger.warning(
            "Legacy WCS compatibility: only one of CRPIX1/CRPIX2 is present; "
            "refusing to reconstruct the missing keyword."
        )
        return header

    # ------------------------------------------------------------------
    # Reconstruct CRPIX1/CRPIX2 from the legacy XCEN/YCEN definitions.
    #
    # XCEN = CRVAL1
    #        + CDELT1*cos(a) * ((NAXIS1+1)/2 - CRPIX1)
    #        - CDELT2*sin(a) * ((NAXIS2+1)/2 - CRPIX2)
    #
    # YCEN = CRVAL2
    #        + CDELT1*sin(a) * ((NAXIS1+1)/2 - CRPIX1)
    #        + CDELT2*cos(a) * ((NAXIS2+1)/2 - CRPIX2)
    # ------------------------------------------------------------------
    naxis1 = float(header["NAXIS1"])
    naxis2 = float(header["NAXIS2"])

    xcen = float(header["XCEN"])
    ycen = float(header["YCEN"])

    cdelt1 = float(header["CDELT1"])
    cdelt2 = float(header["CDELT2"])

    crval1 = float(header["CRVAL1"])
    crval2 = float(header["CRVAL2"])

    crota2 = float(header["CROTA2"])

    angle = np.deg2rad(crota2)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    y1 = (
            xcen
            - crval1
            - cdelt1 * cos_a * (naxis1 + 1.0) / 2.0
            + cdelt2 * sin_a * (naxis2 + 1.0) / 2.0
    )

    y2 = (
            ycen
            - crval2
            - cdelt1 * sin_a * (naxis1 + 1.0) / 2.0
            - cdelt2 * cos_a * (naxis2 + 1.0) / 2.0
    )

    a1 = -cdelt1 * cos_a
    a2 = -cdelt1 * sin_a
    b1 = cdelt2 * sin_a
    b2 = -cdelt2 * cos_a

    determinant = a1 * b2 - a2 * b1

    if not np.isfinite(determinant) or np.isclose(determinant, 0.0):
        raise ValueError(
            "Cannot reconstruct CRPIX1/CRPIX2 from legacy XCEN/YCEN "
            "metadata: invalid or singular geometry "
            f"(determinant={determinant!r})."
        )

    crpix1 = (y1 * b2 - y2 * b1) / determinant
    crpix2 = (a1 * y2 - a2 * y1) / determinant

    if not np.isfinite(crpix1) or not np.isfinite(crpix2):
        raise ValueError(
            "Cannot reconstruct CRPIX1/CRPIX2 from legacy XCEN/YCEN "
            "metadata: obtained invalid values "
            f"({crpix1!r}, {crpix2!r})."
        )

    header["CRPIX1"] = (
        crpix1,
        "[px] Reconstructed from XCEN/YCEN+XSCALE/YSCALE",
    )
    header["CRPIX2"] = (
        crpix2,
        "[px] Reconstructed from XCEN/YCEN+XSCALE/YSCALE",
    )

    logger.warning(
        f"Legacy WCS compatibility: reconstructed CRPIX1={crpix1:.6f} and "
        f"CRPIX2={crpix2:.6f} from XCEN/YCEN + XSCALE/YSCALE.",
    )

    return header
