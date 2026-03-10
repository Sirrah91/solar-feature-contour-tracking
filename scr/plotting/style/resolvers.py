import numpy as np
from typing import Callable

from scr.plotting.types import ContourGroup


def default_style_resolver() -> Callable[..., dict]:
    """
    Return a resolver that always yields the default ContourGroup style.
    Accepts arbitrary positional arguments (id, id+phase, etc.).
    """
    default = ContourGroup(contours=[]).style

    def resolver(*args) -> dict:
        return default

    return resolver


def track_style_resolver(
        *,
        colors: dict[int, np.ndarray],
        linestyle: str = "-",
        linewidth: float = 1.5,
        alpha: float = 1.0,
) -> Callable[[int], dict]:
    def resolver(track_id: int) -> dict:
        return {
            "color": colors[track_id],
            "linestyle": linestyle,
            "linewidth": linewidth,
            "alpha": alpha,
        }

    return resolver


def sunspot_style_resolver(
        *,
        colors: dict[int, np.ndarray],
        linestyle: str = "-",
        linewidth: float = 1.5,
        alpha: float = 1.0,
) -> Callable[[int], dict]:
    def resolver(sunspot_id: int) -> dict:
        return {
            "color": colors[sunspot_id],
            "linestyle": linestyle,
            "linewidth": linewidth,
            "alpha": alpha,
        }

    return resolver


def sunspot_phase_style_resolver(
        *,
        colors: dict[int, np.ndarray],
        linestyles: dict[str, str],
        linewidth: float = 1.5,
        alpha: float = 1.0,
) -> Callable[[int, str], dict]:
    """
    Build a style resolver for (sunspot_id, phase).
    """

    def resolver(sunspot_id: int, phase: str) -> dict:
        return {
            "color": colors[sunspot_id],
            "linestyle": linestyles.get(phase, "-"),
            "linewidth": linewidth,
            "alpha": alpha,
        }

    return resolver
