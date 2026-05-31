"""
Regression-derived field thresholds computed from sliding-window logistic regression.

All values are computed via estimate_regime_baseline over defined inclination
windows. Results are cached per (phase, data_dir) so repeated calls within a
session are free.
"""
from dataclasses import dataclass
from functools import lru_cache
from os import path
import warnings

import pandas as pd

from scr.convective_regimes.analysis.regression import analyse_regression_with_auc_loss
from scr.convective_regimes.core.baselines import estimate_regime_baseline
from scr.convective_regimes.core.models import BaselineEstimate
from scr.convective_regimes.io.filenames import regression_filename
from scr.convective_regimes.settings import DATA_DIR
from scr.convective_regimes.utils.types_alias import SunspotPhase, FilterMode


# ---------------------------------------------------------------------------
# Inclination windows used to extract scalar thresholds from regression curves.
# These define WHERE on the gamma axis the baseline is averaged.
# Shared between compute_thresholds and plot_regression so both use
# identical windows.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BaselineWindow:
    """Inclination window for baseline estimation."""
    gamma_min: float
    gamma_max: float


BASELINE_WINDOWS: dict[tuple[FilterMode, str], BaselineWindow] = {
    ("sunspots", "PQ"): BaselineWindow(gamma_min=60.0, gamma_max=85.0),
    ("sunspots", "UP"): BaselineWindow(gamma_min=5.0, gamma_max=50.0),
    ("pores", "PQ"): BaselineWindow(gamma_min=5.0, gamma_max=40.0),
    ("pores", "UP"): BaselineWindow(gamma_min=5.0, gamma_max=40.0),
}

_INVALID_ESTIMATE = BaselineEstimate(
    mean=float("nan"), std=float("nan"),
    gamma_min=float("nan"), gamma_max=float("nan"),
    n_points=0,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RegressionThresholds:
    """
    All regression-derived field thresholds for one phase.

    Each field is a BaselineEstimate; use .mean for the scalar threshold,
    .std for its uncertainty, and .is_valid to guard against failed fits.

    Naming convention: {object}_{boundary}_{component}
      object     : sunspot | pore
      boundary   : pq (outer, Ic=0.9) | up (inner, Ic=0.5)
      component  : bver | bhor
    """
    sunspot_pq_bhor: BaselineEstimate
    sunspot_pq_bver: BaselineEstimate
    sunspot_up_bver: BaselineEstimate
    sunspot_up_bhor: BaselineEstimate
    pore_pq_bver: BaselineEstimate
    pore_pq_bhor: BaselineEstimate
    pore_up_bver: BaselineEstimate
    pore_up_bhor: BaselineEstimate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_augmented(
        data_dir: str,
        object_type: FilterMode,
        region: str,
        phase: SunspotPhase,
) -> pd.DataFrame | None:
    """
    Load and augment a regression parquet file.

    Returns None with a warning if the file does not exist,
    so callers can degrade gracefully rather than crash.
    """
    filepath = regression_filename(
        data_dir=data_dir,
        object_type=object_type,
        phase=phase,
        region=region,
    )

    if not path.exists(filepath):
        warnings.warn(
            f"Regression file not found, thresholds will be invalid: {filepath}",
            stacklevel=3,
        )
        return None
    return analyse_regression_with_auc_loss(pd.read_parquet(filepath))


def _estimate(
        results: pd.DataFrame | None,
        component: str,
        key: tuple[FilterMode, str],
) -> BaselineEstimate:
    """
    Extract one scalar baseline.

    Returns _INVALID_ESTIMATE if results is None (file was missing)
    or if the window key is not defined.
    """
    if results is None:
        return _INVALID_ESTIMATE

    window = BASELINE_WINDOWS.get(key)
    if window is None:
        return _INVALID_ESTIMATE

    loss_this = f"loss_{component}"
    loss_other = "loss_hor" if component == "ver" else "loss_ver"

    return estimate_regime_baseline(
        b_component=results[f"B{component}_from_combined"].to_numpy(),
        gamma=results["gamma_center"].to_numpy(),
        loss_of_this_component=results[loss_this].to_numpy(),
        loss_of_other_component=results[loss_other].to_numpy(),
        gamma_min=window.gamma_min,
        gamma_max=window.gamma_max,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def compute_thresholds(
        phase: SunspotPhase,
        data_dir: str = DATA_DIR,
) -> RegressionThresholds:
    """
    Compute all regression-derived thresholds for the given phase.

    Cached per (phase, data_dir)  subsequent calls with the same arguments
    return the same object instantly.

    If a regression file is missing, the corresponding estimates are marked
    invalid rather than raising. A warning is emitted for each missing file.

    Parameters
    ----------
    phase : "all", "forming", "stable", or "decaying"
    data_dir : directory containing the regression parquet files
    """
    sunspot_pq = _load_augmented(data_dir, "sunspots", "PQ", phase)
    sunspot_up = _load_augmented(data_dir, "sunspots", "UP", phase)
    pore_pq = _load_augmented(data_dir, "pores", "PQ", phase)
    pore_up = _load_augmented(data_dir, "pores", "UP", phase)

    return RegressionThresholds(
        sunspot_pq_bhor=_estimate(sunspot_pq, "hor", ("sunspots", "PQ")),
        sunspot_pq_bver=_estimate(sunspot_pq, "ver", ("sunspots", "PQ")),
        sunspot_up_bver=_estimate(sunspot_up, "ver", ("sunspots", "UP")),
        sunspot_up_bhor=_estimate(sunspot_up, "hor", ("sunspots", "UP")),
        pore_pq_bver=_estimate(pore_pq, "ver", ("pores", "PQ")),
        pore_pq_bhor=_estimate(pore_pq, "hor", ("pores", "PQ")),
        pore_up_bver=_estimate(pore_up, "ver", ("pores", "UP")),
        pore_up_bhor=_estimate(pore_up, "hor", ("pores", "UP")),
    )
