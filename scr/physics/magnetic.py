import numpy as np
from copy import deepcopy


def compute_Bhor(
        *,
        Bp: np.ndarray,
        Bt: np.ndarray
) -> np.ndarray:
    return np.sqrt((np.square(Bp) + np.square(Bt)))


def compute_Binc(
        *,
        Br: np.ndarray,
        Bhor: np.ndarray
) -> np.ndarray:
    return np.arctan2(Bhor, Br)


def compute_Bamp(
        *,
        Br: np.ndarray,
        Bhor: np.ndarray
) -> np.ndarray:
    return np.sqrt((np.square(Br) + np.square(Bhor)))


def compute_unsigned_inclination(
        *,
        signed_inclination: np.ndarray,
) -> np.ndarray:
    mask = signed_inclination > np.pi / 2.0
    unsigned_inclination = deepcopy(signed_inclination)

    unsigned_inclination[mask] = np.pi - unsigned_inclination[mask]  # remove sign

    return unsigned_inclination
