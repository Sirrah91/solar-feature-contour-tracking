import matplotlib
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_me(
        x: np.ndarray | list,
        *args,
        backend: str = "TkAgg",
        fig_axis_tuple: tuple[Figure, Axes] | None = None,
        **kwargs
) -> tuple[Figure, Axes]:
    """
    Plots data using the specified backend.

    Parameters:
    - x: np.ndarray or list, the x-axis data or matrix to plot.
    - *args: Additional positional arguments accepted by matplotlib's plot functions.
    - backend: str, optional (default="TkAgg"), the matplotlib backend to use.
               This parameter allows you to choose the backend dynamically.
               Supported backends: "Agg", "TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg", etc.
    - **kwargs: Additional keyword arguments accepted by matplotlib's plot functions.

    Returns:
    - fig, axis: tuple containing the matplotlib Figure and Axes objects of the plot.

    Notes:
    - This function dynamically sets the matplotlib backend based on the "backend" parameter.
    - The "backend" parameter defaults to "TkAgg", suitable for interactive plotting.
    - If "backend" is set to "Agg", the function generates plots without displaying them (non-interactive mode).
    - Supported backends may vary depending on the matplotlib installation.
    """
    matplotlib.use(backend=backend)

    interactive_backends = matplotlib.backends.backend_registry.list_builtin(
        matplotlib.backends.BackendFilter.INTERACTIVE
    )

    if fig_axis_tuple is None:
        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    else:
        fig, axis = fig_axis_tuple

    x = np.squeeze(x)
    if np.ndim(x) == 0:
        x = np.reshape(x, (1,))

    if np.ndim(x) == 1:  # line plot
        # Is the first arg y axis?
        if len(args) and isinstance(args[0], np.ndarray | list) and np.size(x) in np.shape(args[0]):
            y = np.squeeze(args[0])
            try:
                axis.plot(x, y, *args[1:], **kwargs)
            except ValueError:
                axis.plot(x, np.transpose(y), *args[1:], **kwargs)
        else:
            axis.plot(x, *args, **kwargs)

        """
        axis.spines["left"].set_position("zero")
        axis.spines["bottom"].set_position("zero")
        axis.spines["right"].set_color("none")
        axis.spines["top"].set_color("none")
        axis.xaxis.set_ticks_position("bottom")
        axis.yaxis.set_ticks_position("left")
        """

    else:  # x is a matrix to plot
        y_max, x_max = np.shape(x)
        im = axis.imshow(x, *args, origin="lower", extent=(0., float(x_max), 0., float(y_max)), aspect="auto", **kwargs)

        divider = make_axes_locatable(axis)
        cax = divider.append_axes(position="right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    if backend.lower() in interactive_backends:
        try:
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        except Exception:
            pass  # Some environments (or backends) don't support window resizing

    plt.tight_layout()

    if backend.lower() in interactive_backends:
        plt.show(block=False)  # Always non-blocking in interactive mode

    return fig, axis
