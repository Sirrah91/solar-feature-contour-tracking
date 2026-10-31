from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SegmentedModel(Protocol):
    """Any piecewise model that exposes breakpoint locations."""
    fit_breaks: np.ndarray


@dataclass(slots=True)
class TransitionRegion:
    """Represents a transition interval in inclination space."""

    start_deg: float
    end_deg: float
    model: SegmentedModel | None = None

    @property
    def width_deg(self) -> float:
        return self.end_deg - self.start_deg

    @property
    def center_deg(self) -> float:
        return 0.5 * (self.start_deg + self.end_deg)


@dataclass(slots=True)
class BaselineEstimate:
    """Weighted regime baseline estimate."""

    mean: float
    std: float
    gamma_min: float
    gamma_max: float
    n_points: int

    @property
    def is_valid(self) -> bool:
        return self.n_points > 0 and np.isfinite(self.mean)


@dataclass(slots=True)
class RegressionResult:
    """Combined regression analysis result."""

    gamma_centers: np.ndarray

    bver: np.ndarray
    bhor: np.ndarray

    loss_ver: np.ndarray
    loss_hor: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            len(self.gamma_centers),
            len(self.bver),
            len(self.bhor),
            len(self.loss_ver),
            len(self.loss_hor),
        }
        if len(lengths) > 1:
            raise ValueError(
                f"All arrays in RegressionResult must have equal length, got {lengths}"
            )


# slots=True is intentionally omitted here because cached_property
# requires a writable instance __dict__, which slots disables.
@dataclass
class ProbabilityMap2D:
    """2D probability/count map."""

    probability: np.ndarray
    counts: np.ndarray

    x_bins: np.ndarray
    y_bins: np.ndarray

    @cached_property
    def x_centers(self) -> np.ndarray:
        return 0.5 * (self.x_bins[:-1] + self.x_bins[1:])

    @cached_property
    def y_centers(self) -> np.ndarray:
        return 0.5 * (self.y_bins[:-1] + self.y_bins[1:])
