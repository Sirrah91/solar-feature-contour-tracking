import numpy as np

from scr.statistics.computation.topology import (
    compute_fractal_dimension,
    compute_components_and_holes
)
from scr.statistics.computation.lengths import (
    compute_lengths,
    compute_spherical_lengths,
)
from scr.statistics.computation.flux_boundary import compute_flux_border
from scr.geometry.contours.sampling import sample_map_at_contour, calc_arc_lengths
from scr.utils.types_alias import Contour, Contours


def compute_contour_geometric_stats(
        *,
        contours: Contours,
        total_contour: Contour,
        lon2D: np.ndarray,
        lat2D: np.ndarray,
        rsun: float,
) -> dict:
    """
    Fractal, Topology, and Length stats.
    """

    # ------------------------
    # Sampling & arc lengths
    # ------------------------
    arc_lengths = [
        calc_arc_lengths(c, lon2d=lon2D, lat2d=lat2D, rsun=rsun)
        for c in contours
    ]

    # ------------------------
    # Topology
    # ------------------------
    fractal_stats = compute_fractal_dimension(
        contours=contours,
        total_contour=total_contour,
        projection_weights=None,
    )

    comp_stats = compute_components_and_holes(
        contours=contours
    )

    # ------------------------
    # Lengths
    # ------------------------
    length_stats = compute_lengths(
        contours=contours,
        total_length=True,
        projection_weights=None,
    )

    length_corr_stats = compute_spherical_lengths(
        arc_lengths=arc_lengths,
        total_length=True,
        projection_weights=None,
    )

    return {
        "fractal-dimension": fractal_stats,
        "topology": comp_stats,
        "lengths_raw": length_stats,
        "lengths_corr": length_corr_stats,
        "_arc_lengths": arc_lengths  # Keep for flux calculation
    }


def compute_contour_intensity_stats(
        *,
        contours: Contours,
        image: np.ndarray,
        inv_mu2D: np.ndarray,
        arc_lengths: list[np.ndarray],
) -> dict:
    """
    Compute all contour-based border flux statistics for a region.
    """

    # ------------------------
    # Sampling & arc lengths
    # ------------------------
    inv_mu2d_samples = [
        sample_map_at_contour(contour=c, data_map=inv_mu2D, interp=True)
        for c in contours
    ]

    field_samples = [
        sample_map_at_contour(contour=c, data_map=image, interp=True)
        for c in contours
    ]

    # Pre-calculate weighted length for corr flux
    length_correction = [
        arc_length * inv_mu2d_sample
        for arc_length, inv_mu2d_sample in zip(arc_lengths, inv_mu2d_samples)
    ]

    # ------------------------
    # Border flux
    # ------------------------
    flux_border_raw = compute_flux_border(
        field1d=field_samples,
        projection_weights=None,
    )

    # --- Corrected border flux ---
    flux_border_corr = compute_flux_border(
        field1d=field_samples,
        projection_weights=length_correction,
    )

    return {
        "flux-border_raw": flux_border_raw,
        "flux-border_corr": flux_border_corr,
    }
