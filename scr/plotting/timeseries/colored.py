import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from typing import Sequence

from scr.plotting.generic.segments import make_colored_segments


def add_colored_timeseries(
        ax: Axes,
        x: Sequence[float],
        y: Sequence[float],
        phases: Sequence[str],
        phase_colors: dict[str, str],
        linestyle: str = "-",
        linewidth: float = 2,
        segment_kwargs: dict | None = None,
) -> LineCollection:
    """
    Add a continuous phase-coloured timeseries to an axis.
    """
    lc = make_colored_segments(
        x=x,
        y=y,
        color_labels=phases,
        colors=phase_colors,
        linestyle=linestyle,
        linewidth=linewidth,
        segment_kwargs=segment_kwargs,
    )
    lc.set_transform(ax.transData)
    ax.add_collection(lc)

    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    ax.autoscale_view(scalex=False, scaley=True)

    lc.set_zorder(3)

    return lc
