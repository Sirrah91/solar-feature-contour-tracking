import numpy as np

from scr.convective_regimes.core.models import TransitionRegion
from scr.statistics.segments.simple_pwlf import piecewise_linear_fit
from scr.config.numerics import RND_SEED


def fit_transition_region(
        *,
        gamma_centers: np.ndarray,
        loss_ver: np.ndarray,
        loss_hor: np.ndarray,
        n_segments: int = 3,
        seed: int = RND_SEED,
) -> TransitionRegion:
    """
    Estimate the inclination transition interval from predictive dominance.

    Fits a piecewise linear model to the horizontal-component alpha curve
    and reads the two interior breakpoints as the transition boundaries.
    """
    total_loss = loss_ver + loss_hor

    alphas_hor = np.where(
        total_loss > 0,
        loss_ver / total_loss,
        0.5,
    )

    model = piecewise_linear_fit(
        gamma_centers,
        alphas_hor,
        n_segments=n_segments,
        seed=seed,
    )

    start, end = model.fit_breaks[1:-1]

    return TransitionRegion(
        start_deg=float(start),
        end_deg=float(end),
        model=model,
    )
