"""Runtime-configurable paths and defaults for convective regime analysis."""
from os import path

DATA_DIR = "/nfsscratch/david/Contours/convective_regimes"
SUNSPOTS_PHASES_DIR_TEMPLATE = "/nfsscratch/david/Contours/sunspots_{sunspot_type}_phases"
FIG_FORMAT = "pdf"


def _default_figure_dir() -> str:
    try:
        from scr.config.paths import PATH_FIGURES
        return path.join(PATH_FIGURES, "paper_plots")
    except ImportError:
        return path.join(DATA_DIR, "figures")


FIGURE_DIR = _default_figure_dir()
