import shutil
import warnings
import matplotlib.pyplot as plt
from contextlib import contextmanager


@contextmanager
def font_style(
        fontsize: int = 12,
        use_tex: bool = True,
        amsmath: bool = True
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
    use_tex : bool, default: True
        Whether to invoke an external LaTeX distribution. If False,
        internal 'mathtext' is used with Computer Modern (cm) fonts.
        Automatically falls back to False if a `latex` executable
        cannot be found on PATH, regardless of the requested value.
    amsmath : bool, default: True
        If True and `use_tex` is True, adds '\\usepackage{amsmath}'
        to the LaTeX preamble.

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

    settings = {
        "text.usetex": use_tex,
        "font.size": fontsize,
        "font.family": "sans-serif",
        "mathtext.fontset": "cm"
    }

    if use_tex and amsmath:
        settings["text.latex.preamble"] = r"\usepackage{amsmath}"

    with plt.rc_context(rc=settings):
        yield
