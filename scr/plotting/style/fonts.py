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
    amsmath : bool, default: True
        If True and `use_tex` is True, adds '\\usepackage{amsmath}'
        to the LaTeX preamble.

    Yields
    ------
    None
        Activates the styling within the `with` block.
    """
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
