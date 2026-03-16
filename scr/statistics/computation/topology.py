import numpy as np

from scr.statistics.computation.kernels.fractal import fractal_kernel
from scr.statistics.computation.aggregator import aggregate_kernel
from scr.geometry.contours.area import contour_signed_area
from scr.utils.types_alias import Contour, Contours, Stat
from scr.utils.filesystem import is_empty


def compute_fractal_dimension(
        *,
        contours: Contours,
        total_contour: Contour | None = None,
        projection_weights: np.ndarray | None = None,
) -> dict:
    """
    Compute fractal dimension of individual contours and the total contour.

    Parameters
    ----------
    contours : list[np.ndarray]=
        List of contours.
    total_contour : np.ndarray | None
        Optional total contour. If not provided, computed from contours.
    projection_weights : np.ndarray | None
        Optional projection correction weights.

    Returns
    -------
    dict
        {"per_object": [fd1, fd2, ...], "global": fd_total}
    """
    if (total_contour is None) and (not is_empty(contours)):
        total_contour = np.vstack(contours)

    def _kernel(contour: np.ndarray) -> float:
        if is_empty(contour):
            return np.nan

        return fractal_kernel(
            values=contour,
            weights=projection_weights,
        )

    return aggregate_kernel(
        kernel=_kernel,
        objects=contours,
        total_object=total_contour,
        listify=False,
    )


def compute_components_and_holes(
        *,
        contours: Contours,
) -> Stat:
    """
    Compute number of components and holes from signed contour areas.

    Parameters
    ----------
    contours : list of Nx2 arrays

    Returns
    -------
    dict
        {
            "n_components": int,
            "n_holes": int,
        }
    """
    areas = [
        0.0 if len(c) < 3 else contour_signed_area(c)
        for c in contours
    ]

    n_components = sum(a > 0 for a in areas)
    n_holes = sum(a < 0 for a in areas)

    return {
        "n_components": np.int32(n_components),
        "n_holes": np.int32(n_holes),
    }
