import numpy as np
from astropy.io import fits
from glob import glob
from os import path
from typing import Literal

from scr.config.numerics import WP
from scr.utils.types_alias import Headers, Quantity
from scr.physics.magnetic import compute_Bhor, compute_Binc


def load_image(
        filename: str,
        quantity: Quantity | Literal["Binc"] | int
) -> np.ndarray:
    if quantity == "Bhor":
        Bp = np.asarray(fits.getdata(filename, "Bp", memmap=True), dtype=np.float64)[0]
        Bt = np.asarray(fits.getdata(filename, "Bt", memmap=True), dtype=np.float64)[0]
        return compute_Bhor(Bp=Bp, Bt=Bt).astype(WP)
    if quantity == "Binc":
        Bp = np.asarray(fits.getdata(filename, "Bp", memmap=True), dtype=np.float64)[0]
        Bt = np.asarray(fits.getdata(filename, "Bt", memmap=True), dtype=np.float64)[0]
        Br = np.asarray(fits.getdata(filename, "Br", memmap=True), dtype=np.float64)[0]
        Bhor = compute_Bhor(Bp=Bp, Bt=Bt)

        return compute_Binc(Br=Br, Bhor=Bhor).astype(WP)

    if quantity == "Bver":
        quantity = "Br"
    return np.asarray(fits.getdata(filename, quantity, memmap=True), dtype=WP)[0]


def load_fits_headers(
        fits_dir_or_filename_list: str | list[str],
        header_index: int = 0,
        regex: str = "*"
) -> Headers:
    if isinstance(fits_dir_or_filename_list, str) and path.isdir(fits_dir_or_filename_list):
        fits_all = sorted(glob(path.join(fits_dir_or_filename_list, regex)))
    else:
        fits_all = fits_dir_or_filename_list

    headers = [fits.getheader(fits_file, header_index, memmap=True) for fits_file in fits_all]

    return headers
