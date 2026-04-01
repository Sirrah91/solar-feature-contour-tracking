from astropy.io import fits
import numpy as np
from glob import glob
from os import path
import re
import warnings
from typing import Literal

from scr.geometry.solar.projection import pixel_to_lonlat
from scr.geometry.wcs.header import fill_header_for_wcs
from scr.physics.magnetic import compute_Bhor, compute_Binc, compute_Bamp
from scr.plotting.generic.hist import plot_pdfs
from scr.devtools.plotting import plot_me

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def read_cotemporal_fits(filename: str, check_uniqueness: bool = False) -> dict:
    fits_folder, fits_name = path.split(filename)
    match = re.search(r"(?:[\w\-]+_)?(\d+s)(?:\.(\d+))?\.(\d{8}_\d{6}_TAI)", fits_name)

    duration = match.group(1)  # "720s"
    optional_harp_number = match.group(2) or ""  # "123" or ""
    timestamp = match.group(3)  # "20250518_123000_TAI"
    filenames = glob(path.join(fits_folder, f"*{duration}*{optional_harp_number}*{timestamp}*"))

    # Keywords to look for
    keywords = [".continuum.fits", ".field.fits", ".inclination.fits", ".azimuth.fits", ".disambig.fits"]
    fits_names = ["ic", "b", "inc", "azi", "disamb"]

    if check_uniqueness:
        for keyword in keywords:
            if sum(keyword in _filename for _filename in filenames) > 1:
                raise ValueError(f'Fits names are not unique: Multiple files contain the keyword "{keyword}".')

    # Create the dictionary
    result_dict = {}
    for index, keyword in enumerate(keywords):
        # Find the first string containing the keyword
        matching_strings = [s for s in filenames if keyword in s]
        result_dict[f"fits_{fits_names[index]}"] = matching_strings[0] if matching_strings else None

    return result_dict


def data_b2ptr(index, bvec: np.ndarray, disambig: np.ndarray | None = None) -> np.ndarray:
    # Fill the header with necessary keywords for WCS
    index = fill_header_for_wcs(header=index)

    # Check dimensions
    nq, ny, nx = np.shape(bvec)
    if nq != 3 or nx != index["NAXIS1"] or ny != index["NAXIS2"]:
        raise ValueError("Dimension of bvec incorrect")

    if disambig is not None:
        # disambiguate azimuth; add 180 to azimuth for rotated images came from im_patch
        bvec[2, :, :] = disambigue_azimuth(bvec[2, :, :], disambig, method="random",
                                           k_rot=crota2_to_krot(index.get("CROTA2", 180.)))

    # Convert bvec to B_xi, B_eta, B_zeta
    field = bvec[0, :, :]
    gamma = np.deg2rad(bvec[1, :, :])
    psi = np.deg2rad(bvec[2, :, :])

    b_xi = -field * np.sin(gamma) * np.sin(psi)
    b_eta = field * np.sin(gamma) * np.cos(psi)
    b_zeta = field * np.cos(gamma)

    lon, lat = pixel_to_lonlat(header=index)

    # Get matrix to convert
    b = np.deg2rad(index["CRLT_OBS"])  # b-angle, disk center latitude
    p = np.deg2rad(-index["CROTA2"])  # p-angle, negative of CROTA2

    phi = np.deg2rad(lon)
    lambda_ = np.deg2rad(lat)

    sinb, cosb = np.sin(b), np.cos(b)
    sinp, cosp = np.sin(p), np.cos(p)
    sinphi, cosphi = np.sin(phi), np.cos(phi)
    sinlam, coslam = np.sin(lambda_), np.cos(lambda_)

    k11 = coslam * (sinb * sinp * cosphi + cosp * sinphi) - sinlam * cosb * sinp
    k12 = -coslam * (sinb * cosp * cosphi - sinp * sinphi) + sinlam * cosb * cosp
    k13 = coslam * cosb * cosphi + sinlam * sinb
    k21 = sinlam * (sinb * sinp * cosphi + cosp * sinphi) + coslam * cosb * sinp
    k22 = -sinlam * (sinb * cosp * cosphi - sinp * sinphi) - coslam * cosb * cosp
    k23 = sinlam * cosb * cosphi - coslam * sinb
    k31 = -sinb * sinp * sinphi + cosp * cosphi
    k32 = sinb * cosp * sinphi + sinp * cosphi
    k33 = -cosb * sinphi

    # Output
    bptr = np.zeros_like(bvec)
    bptr[0, :, :] = k31 * b_xi + k32 * b_eta + k33 * b_zeta
    bptr[1, :, :] = k21 * b_xi + k22 * b_eta + k23 * b_zeta
    bptr[2, :, :] = k11 * b_xi + k12 * b_eta + k13 * b_zeta

    return bptr


def crota2_to_krot(crota2: float, crota2_ref: float = 180.) -> int:
    """
    Convert the CROTA2 image rotation angle (in degrees) to an integer number of
    90° counter-clockwise rotations (k_rot) relative to a reference rotation angle.

    The standard reference corresponds to CROTA2 = 180°, where zero azimuth points down
    (toward the +Y pixel direction in image coordinates).

    Parameters
    ----------
    crota2 : float
        Image rotation angle in degrees, typically from the FITS header keyword CROTA2.
    crota2_ref : float, optional
        Reference CROTA2 angle corresponding to zero rotation (k_rot = 0).
        Default is 180° (zero azimuth pointing down).

    Returns
    -------
    int
        Number of 90° counter-clockwise rotations relative to crota2_ref,
        i.e. k_rot ∈ {0, 1, 2, 3}. For example:
          - k_rot = 0: crota2 == crota2_ref (zero azimuth down)
          - k_rot = 1: 90° CCW rotation
          - k_rot = 2: 180° CCW rotation
          - k_rot = 3: 270° CCW rotation

    Notes
    -----
    The function normalizes crota2 to [0°, 360°), computes the offset from crota2_ref,
    rounds to the nearest multiple of 90°, and wraps into {0, 1, 2, 3}.
    """
    crota2_norm = crota2 % 360.
    delta = (crota2_norm - crota2_ref) % 360.
    k_rot = int(round(delta / 90.)) % 4

    return k_rot


def parse_disambig_method(_method: Literal[0, 1, 2, "potential_acute", "random", "radial_acute"],
                          return_string: bool = False) -> Literal[0, 1, 2, "potential_acute", "random", "radial_acute"]:
    """Convert method input (string or int) into a valid integer index."""
    methods = {0: "potential_acute",
               1: "random",
               2: "radial_acute",
               "default": 1}

    methods_reverse = {value: key for key, value in methods.items() if isinstance(key, int)}
    options = ", ".join(f'"{name}" == {number}' for name, number in methods_reverse.items())

    original_method = _method  # Store original value for warning message

    # Convert string to corresponding integer
    if isinstance(_method, str):
        _method = methods_reverse.get(_method, None)

    # Validate method (ensure it's one of the allowed int keys)
    if not isinstance(_method, int) or _method not in methods:
        warnings.warn(f'Invalid disambiguation method "{original_method}".\n'
                      f'\tValid string or integer options: {options}. Defaulting to "{methods[methods["default"]]}".')
        _method = methods["default"]

    return methods[_method] if return_string else _method


def disambigue_azimuth(
        azimuth: np.ndarray,
        disambig: np.ndarray,
        method: Literal[0, 1, 2, "potential_acute", "random", "radial_acute"] = "random",
        k_rot: int = 0
) -> np.ndarray:
    """
    Apply disambiguation to ambiguous azimuth angles in [0°, 180°) based on the specified method,
    and adjust for image rotation in multiples of 90° relative to CROTA2=180° (zero azimuth down).

    Parameters
    ----------
    azimuth : np.ndarray
        Ambiguous azimuth angles in degrees, expected in [0°, 180°).
    disambig : np.ndarray
        Array containing disambiguation flags encoded as integers.
    method : {0, 1, 2, "potential_acute", "random", "radial_acute"}, optional
        Disambiguation method to use. Default is "random".
    k_rot : int, optional
        Number of 90° counter-clockwise rotations relative to CROTA2=180° orientation.
        Default is 0.

    Returns
    -------
    np.ndarray
        Disambiguated azimuth angles in degrees, in the range [0°, 360°),
        adjusted for image rotation.
    """

    def extract_bit(arr: np.ndarray, bit: int) -> np.ndarray:
        """Extract a specific bit from an encoded integer array."""
        # arr >> bit moves the binary representation of arr to the right by "bit" places
        # The bits that are shifted out are lost, e.g. 21 >> 2 (10101 -> 101.01 -> 101)
        # Similarly arr << bit will add "bit" zeros at the end (0b10101 << 2 -> 0b1010100)
        # arr & N performs bit-wise and, e.g. 0b10101 & 0b1101 -> 0b00101
        return ((arr.astype(int) >> bit) & 1).astype(float)

    disambig_matrix = extract_bit(disambig, bit=parse_disambig_method(method))

    # Disambiguation correction: add 180° to pixels flagged by disambig_matrix
    correction = disambig_matrix * 180.

    # Apply correction then rotate azimuth forward by k_rot * 90 deg CCW
    azimuth_corrected = azimuth + correction

    # The final azimuth is in the image coordinate system with zero azimuth pointing down (+Y axis).
    # In array (i,j) indexing, this corresponds to upward direction. Required by `data_b2ptr` function.
    azimuth_rotated = azimuth_corrected + k_rot * 90.

    return azimuth_rotated % 360.


dcon_dirs = sorted(glob("/nfsscratch/david/NN/results/*"))

dcon_dir = dcon_dirs[-1]
dcon_files = sorted(glob(path.join(dcon_dir, "*.fits")))

dcon_file = dcon_files[len(dcon_files)//2]
hmi_file = dcon_file \
    .replace("/results/", "/data/SDO_HMI/") \
    .replace(".proc_", ".b_") \
    .replace(".ibptr.", ".field.") \
    .replace("_dconANN.", ".")

hmi_fits = read_cotemporal_fits(hmi_file)
hdul = fits.open(dcon_file)

plot_me(hdul["Ic"].data[0])

index = fits.getheader(hmi_fits["fits_b"], 1)
bvec = np.array([fits.getdata(hmi_fits["fits_b"], 1),
                 fits.getdata(hmi_fits["fits_inc"], 1),
                 fits.getdata(hmi_fits["fits_azi"], 1)])
disambig = fits.getdata(hmi_fits["fits_disamb"], 1)

bptr = data_b2ptr(index, bvec, disambig)

Bhor_dcon = compute_Bhor(Bp=hdul["Bp"].data[0], Bt=hdul["Bt"].data[0])
Br_dcon = hdul["Br"].data[0]

Bhor_hmi = compute_Bhor(Bp=bptr[0], Bt=bptr[1])
Br_hmi = bptr[2]

Br_hmi, Bhor_hmi = np.rot90(Br_hmi, 2), np.rot90(Bhor_hmi, 2)

B_dcon = compute_Bamp(Br_dcon, Bhor_dcon)
B_hmi = compute_Bamp(Br_hmi, Bhor_hmi)

fig, ax = plot_me(Bhor_hmi)
im = ax.images[0]
im.set_clim(np.nanmin(Bhor_hmi), np.nanmax(Bhor_hmi))
fig, ax = plot_me(Bhor_dcon)
im = ax.images[0]
im.set_clim(np.nanmin(Bhor_hmi), np.nanmax(Bhor_hmi))

gamma_dcon = np.rad2deg(compute_Binc(np.abs(Br_dcon), Bhor_dcon))
gamma_hmi = np.rad2deg(compute_Binc(np.abs(Br_hmi), Bhor_hmi))

gamma_dcon[B_dcon < 500] = np.nan
gamma_hmi[B_hmi < 500] = np.nan

plot_me(gamma_hmi)
plot_me(gamma_dcon)

fig, ax = plt.subplots(figsize=(8, 6))
plot_pdfs(
    ax=ax,
    datasets=[gamma_hmi, gamma_dcon],
    labels=["HMI", "CNN"],
    colors=["red", "blue"],
    linestyles=["-", "-"],
)
ax.legend()
ax.set_title("LRF of both from `hmi_b2ptr.pro`")
ax.set_ylabel(r"PDF (%)")
ax.set_xlabel(r"Inclination from vertical (deg)")
plt.show(block=False)
fig.savefig("Inc_HMI_vs_CNN.pdf")

hdul.close()

CNN = sorted(glob("/solardata/archiv/jurcak/data_Iulia/Bxyz_CNN/*Bx*"))[59]
HMI = sorted(glob("/solardata/archiv/jurcak/data_Iulia/Bxyz_HMI/*Bx*"))[58]
SSIA = sorted(glob("/solardata/archiv/jurcak/data_Iulia/Bxyz_SSIA/*Bx*"))[57]

def load_data(what):
    with fits.open(what) as hdul:
        Bx = hdul[0].data
    with fits.open(what.replace("_Bx.", "_By.")) as hdul:
        By = hdul[0].data
    with fits.open(what.replace("_Bx.", "_Bz.")) as hdul:
        Bz = hdul[0].data

    return [Bx, By, Bz]

CNN = load_data(CNN)
HMI = load_data(HMI)
SSIA = load_data(SSIA)

Bhor_CNN = np.sqrt(CNN[0]**2 + CNN[1] ** 2)
Bhor_HMI = np.sqrt(HMI[0]**2 + HMI[1] ** 2)
Bhor_SSIA = np.sqrt(SSIA[0]**2 + SSIA[1] ** 2)

gamma_CNN = np.rad2deg(compute_Binc(np.abs(CNN[2]), Bhor_CNN))
gamma_HMI = np.rad2deg(compute_Binc(np.abs(HMI[2]), Bhor_HMI))
gamma_SSIA = np.rad2deg(compute_Binc(np.abs(SSIA[2]), Bhor_SSIA))


B_CNN = compute_Bamp(CNN[2], Bhor_CNN)
B_HMI = compute_Bamp(HMI[2], Bhor_HMI)
B_SSIA = compute_Bamp(SSIA[2], Bhor_SSIA)

gamma_CNN[B_CNN < 500] = np.nan
gamma_HMI[B_HMI < 500] = np.nan
gamma_SSIA[B_SSIA < 500] = np.nan

fig, ax = plt.subplots(figsize=(8, 6))
plot_pdfs(
    ax=ax,
    datasets=[gamma_HMI, gamma_CNN, gamma_SSIA],
    labels=["HMI", "CNN", "SSIA"],
    colors=["red", "blue", "green"],
    linestyles=["-", "-", "-"],
)
ax.legend()
ax.set_title("LRF from Iulia")
ax.set_ylabel(r"PDF (%)")
ax.set_xlabel(r"Inclination from vertical (deg)")
plt.show(block=False)
fig.savefig("Inc_HMI_vs_CNN_vs_SSIA.pdf")
