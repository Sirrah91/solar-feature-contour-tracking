from typing import NamedTuple


class LineSpec(NamedTuple):
    """Visual specification for a single reference or iso-contour line.

    Parameters
    ----------
    level : data-coordinate value at which the line is drawn
    color : matplotlib color string
    linestyle : matplotlib linestyle string, e.g. "-", "--", ":"
    linewidth : line width in points (default 2.0)
    """
    level: float
    color: str
    linestyle: str
    linewidth: float = 2.0
