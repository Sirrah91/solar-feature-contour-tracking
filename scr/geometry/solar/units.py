import numpy as np
import astropy.units as u


def arcsec_to_Mm(
        arcsec: float,
        distanceAU: float = 1.,
        center_distance: bool = True
) -> float:
    fun = np.tan if center_distance else np.sin
    return 2. * u.AU.to(u.Mm, distanceAU) * fun(u.arcsec.to(u.rad, arcsec) / 2.)


def pixelarea_to_Mm2(px_area: float) -> float:
    return px_area * arcsec_to_Mm(0.319978) * arcsec_to_Mm(0.29714)
