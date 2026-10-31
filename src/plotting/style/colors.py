import numpy as np
from matplotlib import colormaps
from typing import Iterable


def sample_cmap(
        cmap: str,
        n: int,
        *,
        start: float = 0.0,
        stop: float = 1.0,
) -> np.ndarray:
    """
    Sample `n` distinct colours from a matplotlib colormap.

    Returns
    -------
    np.ndarray
        Array of shape (n, 4) in RGBA.
    """
    if n <= 0:
        return np.empty((0, 4))

    cmap_obj = colormaps[cmap]
    values = np.linspace(start, stop, n)
    return cmap_obj(values)


def id_colors(
        ids: Iterable[int],
        cmap: str
) -> dict[int, np.ndarray]:
    ids = list(ids)
    return dict(zip(ids, sample_cmap(cmap, len(ids))))
