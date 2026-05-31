import numpy as np

from scr.convective_regimes.core.models import BaselineEstimate
from scr.statistics.numerics.weighted import weighted_average, weighted_std


def estimate_regime_baseline(
        *,
        b_component: np.ndarray,
        gamma: np.ndarray,
        loss_of_this_component: np.ndarray,
        loss_of_other_component: np.ndarray,
        gamma_min: float,
        gamma_max: float,
) -> BaselineEstimate:
    """
    Estimate weighted baseline field strength over an inclination segment.

    The weight of each point reflects how dominant this component is:
    high loss when the *other* component is dropped → this component matters more.
    """
    total_loss = loss_of_this_component + loss_of_other_component

    raw_weights = np.where(total_loss > 0, loss_of_this_component / total_loss, 0.5)

    mask = (
            (gamma >= gamma_min)
            & (gamma <= gamma_max)
            & np.isfinite(b_component)
    )

    values = b_component[mask]
    weights = raw_weights[mask]

    if values.size == 0:
        return BaselineEstimate(
            mean=np.nan,
            std=np.nan,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_points=0,
        )

    mean = weighted_average(array=values, weights=weights)
    std = weighted_std(array=values, mean=mean, weights=weights)

    return BaselineEstimate(
        mean=mean,
        std=std,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        n_points=int(values.size),
    )
