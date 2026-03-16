from dataclasses import dataclass, field
import numpy as np
from typing import Callable

from scr.utils.types_alias import Contours


@dataclass(frozen=True)
class ContourGroup:
    """
    A styled group of contours to be plotted together.
    """
    contours: Contours
    style: dict = field(default_factory=lambda: {
        "color": "red",
        "linestyle": "-",
        "linewidth": 1.0,
        "alpha": 1.0,
    })
    label: str | None = None


@dataclass
class OverlaySpec:
    """
    Specification for computing and drawing an overlay
    derived from a support image.
    """
    extractor: Callable[[np.ndarray], list[np.ndarray]]
    matcher: Callable[[np.ndarray, list[np.ndarray]], np.ndarray | None]
    color: str = "blue"
    linewidth: float = 1.0


@dataclass
class PhaseVideoInputs:
    images: list[np.ndarray]
    tracks: dict

    times: np.ndarray
    y_main: np.ndarray
    y_std: np.ndarray
    phases: np.ndarray
    colors: dict[str, str]

    support_images: list[np.ndarray] | None = None
    overlay: OverlaySpec | None = None
