import numpy as np
import plotly.graph_objects as go

from scr.plotting.utils import merge_explicit_kwargs


def plot_volume_3d(
        fig: go.Figure,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
        *,
        name: str = "Volume",
        opacity: float = 0.1,
        surface_count: int = 25,
        colorscale: str = "hot_r",
        isomin: float | None = None,
        isomax: float | None = None,
        volume_kwargs: dict | None = None,
) -> None:
    """
    Plot a 3D isosurface volume on a Plotly Figure.
    """
    volume_kwargs = merge_explicit_kwargs(
        volume_kwargs,
        x=x.ravel(),
        y=y.ravel(),
        z=z.ravel(),
        value=values.ravel(),
        opacity=opacity,
        isomin=isomin if isomin is not None else np.nanmin(values),
        isomax=isomax if isomax is not None else np.nanmax(values),
        surface_count=surface_count,
        colorscale=colorscale,
        name=name,
    )

    fig.add_trace(go.Volume(**volume_kwargs))
