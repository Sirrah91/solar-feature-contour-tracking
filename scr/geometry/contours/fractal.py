import numpy as np

from scr.utils.types_alias import Contour, Mask
from scr.utils.filesystem import is_empty


def fractal_dimension_mask(
        mask: Mask,
        n_scales: int = 10,
        control_plot: bool = False
) -> float:
    """
    Compute the box-counting (Minkowski-Bouligand) fractal dimension
    of a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        2D boolean array where True marks the set whose fractal
        dimension is to be measured.
    n_scales : int, optional
        Number of box sizes sampled logarithmically between the smallest
        and largest meaningful scales. Default is 10.
    control_plot : bool, optional
        If True, produce a loglog plot of box count vs. scale.

    Returns
    -------
    float
        Estimated fractal dimension. Returns NaN if the mask is empty.

    Notes
    -----
    The method partitions the mask into square boxes of size ε and counts
    how many boxes contain at least one True pixel. The slope of
        log N(ε) vs log(1/ε)
    is the box-counting dimension.
    """
    if not np.any(mask):
        return np.nan
    mask = np.asarray(mask) > 0.5

    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")

    H, W = mask.shape
    min_dim = min(H, W)

    # Choose scales: from 1 px up to ~min_dim / 2
    # Logspace in terms of box side length
    scales = np.logspace(0, np.log10(min_dim / 2), num=n_scales)
    scales = np.unique(scales.astype(int))
    scales = scales[scales > 1]

    if len(scales) < 2:
        return float("nan")

    counts = []
    eps_values = []

    for eps in scales:
        # number of boxes in each dimension
        nH = int(np.ceil(H / eps))
        nW = int(np.ceil(W / eps))

        # Count non-empty boxes
        count = 0
        for i in range(nH):
            for j in range(nW):
                patch = mask[i * eps:(i + 1) * eps, j * eps:(j + 1) * eps]
                if patch.any():
                    count += 1

        counts.append(count)
        eps_values.append(eps)

    # Fit slope: log N(ε) vs log 1/ε
    logs_eps = np.log(1.0 / np.array(eps_values))
    logs_N = np.log(np.array(counts))

    coeffs = np.polyfit(logs_eps, logs_N, 1)
    logs_N_fit = np.polyval(coeffs, logs_eps)

    D = float(coeffs[0])

    if control_plot:
        from os import path
        from scr.config.paths import PATH_FIGURES
        from scr.utils.filesystem import check_dir
        from scr.geometry.contours.control_plots import plot_fractal_dimension_control

        fig_outdir = path.join(PATH_FIGURES, "fractal_dimension")
        check_dir(fig_outdir)

        plot_fractal_dimension_control(
            logs_eps=logs_eps,
            logs_N=logs_N,
            logs_N_fit=logs_N_fit,
            fractal_dim=D,
            outfile=path.join(fig_outdir, "box_counting.jpg"),
            use_tex=False,
        )

    return D


def fractal_dimension_contour(
        contour: Contour,
        n_scales: int = 10,
        control_plot: bool = False
) -> float:
    """
    Box-counting fractal dimension computed directly
    from a contour (Nx2 arrays).
    """

    if is_empty(contour) or len(contour) < 2:
        return float("nan")

    x = contour[:, 0]
    y = contour[:, 1]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    L = max(xmax - xmin, ymax - ymin)

    if L == 0:
        return float("nan")

    # scales from large to small
    scales = np.logspace(
        np.log10(L / 2),
        np.log10(L / 100),
        n_scales
    )

    counts = []
    eps_values = []

    for eps in scales:
        ix = np.floor((x - xmin) / eps).astype(int)
        iy = np.floor((y - ymin) / eps).astype(int)

        boxes = np.stack((ix, iy), axis=1)

        n_boxes = len(np.unique(boxes, axis=0))

        if n_boxes > 0:
            counts.append(n_boxes)
            eps_values.append(eps)

    if len(counts) < 2:
        return float("nan")

    logs_eps = np.log(1.0 / np.array(eps_values))
    logs_N = np.log(np.array(counts))

    coeffs = np.polyfit(logs_eps, logs_N, 1)
    logs_N_fit = np.polyval(coeffs, logs_eps)

    D = float(coeffs[0])

    if control_plot:
        from os import path
        from scr.config.paths import PATH_FIGURES
        from scr.utils.filesystem import check_dir
        from scr.geometry.contours.control_plots import plot_fractal_dimension_control

        fig_outdir = path.join(PATH_FIGURES, "fractal_dimension")
        check_dir(fig_outdir)

        plot_fractal_dimension_control(
            logs_eps=logs_eps,
            logs_N=logs_N,
            logs_N_fit=logs_N_fit,
            fractal_dim=D,
            outfile=path.join(fig_outdir, "box_counting.jpg")
        )

    return D
