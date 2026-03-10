from dataclasses import dataclass, field
import numpy as np
from matplotlib.axes import Axes
from typing import Sequence, Callable

from scr.plotting.types import ContourGroup


@dataclass(frozen=True)
class FrameData:
    """
    Data required to render a single plotting scene.
    Pure data container, no plotting logic.
    """
    image: np.ndarray | None = None
    contour_groups: Sequence[ContourGroup] = field(default_factory=tuple)

    image_kwargs: dict | None = None
    contour_kwargs: dict | None = None

    annotations: Callable[[Axes], None] | None = None
