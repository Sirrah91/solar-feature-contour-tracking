from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import Formatter
import matplotlib.pyplot as plt
from typing import Callable


def add_colorbar(
        ax: Axes,
        mappable: ScalarMappable,
        *,
        cbar_kwargs: dict,
        label: str | None = None,
        formatter: Formatter | str | Callable | None = None,
) -> Colorbar:
    """
    Attach a colorbar to an Axes.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**cbar_kwargs)

    cbar = plt.colorbar(mappable, cax=cax)

    if formatter is not None:
        cbar.ax.yaxis.set_major_formatter(formatter)

    if label is not None:
        cbar.set_label(label)

    return cbar
