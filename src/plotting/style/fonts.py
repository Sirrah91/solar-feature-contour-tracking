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
        Matplotlib's internal MathText renderer is used with a font
        matching the selected font_family. Automatically falls back
        to False if a `latex` executable cannot be found on PATH.

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
        preamble = [r"\usepackage{amsmath}"]

        if font_family == "sans-serif":
            preamble.append(r"\usepackage{sfmath}")

        settings["text.latex.preamble"] = "".join(preamble)

    with plt.rc_context(rc=settings):
        yield
