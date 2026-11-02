import shutil
import warnings
import matplotlib.pyplot as plt
from contextlib import contextmanager
from typing import Literal


@contextmanager
def font_style(
        fontsize: int = 12,
        font_family: Literal["serif", "sans-serif"] = "sans-serif",
        use_tex: bool = True,
):
    """
    Context manager to set plot typography and LaTeX rendering.

    Adjusts matplotlib's global rcParams to achieve a consistent font style
    suitable for scientific publications. Can toggle between true LaTeX
    rendering and matplotlib's internal 'mathtext' simulation.

    Parameters
    ----------
    fontsize : int, default: 12
        Base font size in points.
    font_family : str, default "sans-serif"
        Font family to use.
    use_tex : bool, default: True
        Whether to invoke an external LaTeX distribution. If False,
        internal 'mathtext' is used with Computer Modern (cm) fonts.
        Automatically falls back to False if a `latex` executable
        cannot be found on PATH, regardless of the requested value.

    Yields
    ------
    None
        Activates the styling within the `with` block.
    """
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
        settings["text.latex.preamble"] = r"\usepackage{amsmath}"

    with plt.rc_context(rc=settings):
        yield
