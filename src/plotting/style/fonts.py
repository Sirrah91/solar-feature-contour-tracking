import shutil
import warnings
import matplotlib.pyplot as plt
from contextlib import contextmanager
from typing import Literal


def _get_font_settings(
        fontsize: int = 12,
        font_family: Literal["serif", "sans-serif"] = "sans-serif",
        use_tex: bool = True,
) -> dict:
    if use_tex and shutil.which("latex") is None:
        warnings.warn(
            "use_tex=True requested but no 'latex' executable found on "
            "PATH; falling back to mathtext rendering.",
            RuntimeWarning,
            stacklevel=2,
        )
        use_tex = False

    mathtext_fontsets = {
        "serif": "dejavuserif",
        "sans-serif": "dejavusans",
    }

    settings = {
        "text.usetex": use_tex,
        "font.size": fontsize,
        "font.family": font_family,
        "mathtext.fontset": mathtext_fontsets[font_family],
    }

    if use_tex:
        preamble = [r"\usepackage{amsmath}"]

        if font_family == "sans-serif":
            preamble.append(r"\usepackage{sfmath}")

        settings["text.latex.preamble"] = "".join(preamble)

    return settings


@contextmanager
def font_style(
        fontsize: int = 12,
        font_family: Literal["serif", "sans-serif"] = "sans-serif",
        use_tex: bool = True,
):
    """
    Temporarily apply consistent typography and LaTeX rendering settings.

    Updates Matplotlib's global rcParams within the context and automatically
    restores their previous values upon exit.

    Intended primarily for non-interactive plotting, where figures are
    created, configured, and rendered (e.g. saved) within the context.

    For interactive, non-blocking plotting, use set_font_style() instead,
    so that the settings remain active while figures can still be rendered
    or redrawn.

    Parameters
    ----------
    fontsize : int, default: 12
        Base font size in points.

    font_family : {"serif", "sans-serif"}, default: "sans-serif"
        Font family used for plot text and mathematical expressions.

    use_tex : bool, default: True
        Whether to use external LaTeX for text rendering. If False,
        Matplotlib's internal MathText renderer is used instead.
        When enabled, amsmath is loaded automatically. For sans-serif
        fonts, sfmath is also loaded to provide matching mathematical
        typography. If no `latex` executable is found on PATH, a warning
        is issued and MathText rendering is used instead.

    Yields
    ------
    None
        Activates the styling within the context.

    Notes
    -----
    The settings are restored immediately upon leaving the context.
    Consequently, figures that may be rendered or redrawn afterwards
    should use set_font_style() instead.

    For interactive plotting with blocking display, use
    plt.show(block=True). This keeps the context active until all
    figure windows are closed.

    Alternatively, for non-blocking display, fig.canvas.draw() forces
    synchronous rendering while the context is still active:

        fig.canvas.draw()
        plt.show(block=False)

    This ensures that the initial rendering uses the configured settings.
    However, subsequent GUI redraws (e.g. when resizing or uncovering
    a window) may happen after the context exits. For reliable rendering
    throughout the lifetime of interactive figures, use set_font_style().

    Examples
    --------
    Blocking interactive display:

    >>> with font_style(fontsize=20, font_family="serif"):
    ...     fig, ax = plt.subplots()
    ...     ax.set_xlabel(r"$\\text{Pore area}$")
    ...     fig.savefig("pore_area.pdf")
    ...     plt.show(block=True)

    Non-blocking display with synchronous initial rendering:

    >>> with font_style(fontsize=20):
    ...     fig, ax = plt.subplots()
    ...     ax.set_xlabel(r"$\\text{Pore area}$")
    ...     fig.canvas.draw()
    ...     plt.show(block=False)
    """
    with plt.rc_context(rc=_get_font_settings(
            fontsize=fontsize,
            font_family=font_family,
            use_tex=use_tex,
    )):
        yield


def set_font_style(
        fontsize: int = 12,
        font_family: Literal["serif", "sans-serif"] = "sans-serif",
        use_tex: bool = True,
) -> dict:
    """
    Persistently apply consistent typography and LaTeX rendering settings.

    Updates Matplotlib's global rcParams without automatically restoring
    their previous values. The settings remain active until explicitly
    changed or restored.

    Intended primarily for interactive plotting, where figures can be
    rendered or redrawn after the plotting code has finished executing,
    including non-blocking GUI display.

    The previous values of all modified rcParams are returned, allowing
    the original configuration to be restored when the figures are no
    longer needed.

    Parameters
    ----------
    fontsize : int, default: 12
        Base font size in points.

    font_family : {"serif", "sans-serif"}, default: "sans-serif"
        Font family used for plot text and mathematical expressions.

    use_tex : bool, default: True
        Whether to use external LaTeX for text rendering. If False,
        Matplotlib's internal MathText renderer is used instead.
        When enabled, amsmath is loaded automatically. For sans-serif
        fonts, sfmath is also loaded to provide matching mathematical
        typography. If no `latex` executable is found on PATH, a warning
        is issued and MathText rendering is used instead.

    Returns
    -------
    dict
        Previous values of the modified rcParams. Pass this dictionary
        to `plt.rcParams.update()` to restore the previous configuration.

    Notes
    -----
    Matplotlib rcParams are global. The settings therefore affect all
    figures, not just those created after this function is called.

    Restore the previous configuration only after all figures requiring
    these settings have been closed or are no longer being rendered.

    Examples
    --------
    >>> previous_style = set_font_style(fontsize=20)
    >>> fig, ax = plt.subplots()
    >>> ax.set_xlabel(r"$\\text{Pore area}$")
    >>> plt.show(block=False)

    Restore the previous configuration after closing the figures:

    >>> plt.close("all")
    >>> plt.rcParams.update(previous_style)
    """
    settings = _get_font_settings(
        fontsize=fontsize,
        font_family=font_family,
        use_tex=use_tex,
    )

    previous = {
        key: plt.rcParams[key]
        for key in settings
    }

    plt.rcParams.update(settings)

    return previous
