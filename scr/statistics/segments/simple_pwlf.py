import pwlf
import numpy as np

from scr.config.numerics import RND_SEED


def piecewise_linear_fit(
        x: np.ndarray,
        y: np.ndarray,
        *,
        n_segments: int = 2,
        seed: int = RND_SEED
) -> pwlf.PiecewiseLinFit:
    model = pwlf.PiecewiseLinFit(x, y, seed=seed)
    model.fit(n_segments)

    return model
