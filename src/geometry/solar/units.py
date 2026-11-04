import numpy as np
import astropy.units as u


def arcsec_to_Mm(arcsec: float, distance_au: float = 1.0):
    return (distance_au * u.AU * np.sin(arcsec * u.arcsec)).to_value(u.Mm)


def Mm_to_arcsec(Mm: float , distance_au: float = 1.0):
    return np.arcsin(Mm * u.Mm / (distance_au * u.AU)).to_value(u.arcsec)


def pixelarea_to_Mm2(px_area: float) -> float:
    return px_area * arcsec_to_Mm(0.319978) * arcsec_to_Mm(0.29714)
