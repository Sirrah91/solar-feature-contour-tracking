import numpy as np
from matplotlib.collections import LineCollection
from typing import Sequence

from scr.plotting.utils import merge_explicit_kwargs


def make_colored_segments(
        x: Sequence[float],
        y: Sequence[float],
        color_labels: Sequence[str],
        colors: dict[str, str],
        linestyle: str = "-",
        linewidth: float = 2,
        segment_kwargs: dict | None = None,
) -> LineCollection:
    """
    Build a LineCollection with per-segment coloring.

    Segment i is coloured according to color_labels[i].
    """
    if not (len(x) == len(y) == len(color_labels)):
        raise ValueError("x, y and color_labels must have the same length")

    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    seg_cols = [colors[c] for c in color_labels[:-1]]

    segment_kwargs = merge_explicit_kwargs(
        segment_kwargs,
        colors=seg_cols,
        linewidth=linewidth,
        linestyles=linestyle,
    )

    return LineCollection(
        segs,
        **segment_kwargs,
    )
