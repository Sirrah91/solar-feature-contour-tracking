from typing import Callable, Sequence
from matplotlib.axes import Axes

from scr.utils.types_alias import Contour
from scr.plotting.types import ContourGroup


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _centroid(contour: Contour) -> tuple[float, float]:
    """
    Compute centroid of a contour with shape (N, 2) in (row, col).
    Returns (x, y) for plotting.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("Contour must have shape (N, 2)")

    y, x = contour.mean(axis=0)
    return x, y


def _group_color(group: ContourGroup) -> str:
    return group.style.get("color", "red")


# ---------------------------------------------------------------------
# annotation factories
# ---------------------------------------------------------------------

def annotate_contour_groups_labels(
        *,
        fontsize: float = 8,
) -> Callable[[Axes, Sequence[ContourGroup]], None]:
    """
    Annotate each contour with the group label (e.g. sunspot_id).
    """

    def annotate(ax: Axes, groups: Sequence[ContourGroup]) -> None:
        for group in groups:
            if not group.contours or group.label is None:
                continue

            color = _group_color(group)

            # choose the largest contour (by number of points)
            contour = max(group.contours, key=len)
            x, y = _centroid(contour)

            ax.text(
                x, y,
                group.label,
                color=color,
                fontsize=fontsize,
                ha="center",
                va="center",
            )

    return annotate
