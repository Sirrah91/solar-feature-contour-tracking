"""Solar Physics Fundamentals (Constants)."""
import astropy.units as u
from astropy.coordinates import Distance, Angle

#: Reference solar radius in meters
RSUN_REF: Distance = Distance(696_000_000.0 * u.m)

#: Reference solar distance in meters
DSUN_REF: Distance = Distance(1.0 * u.AU)

#: Reference apparent solar radius at 1 AU.
RSUN_REF_ANGULAR: Angle = Angle(960.0 * u.arcsec)
