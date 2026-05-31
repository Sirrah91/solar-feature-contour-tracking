import numpy as np
from math import comb
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
        show_pdfs: bool = False,
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

    # Functional PDF Projection Logic
    if show_pdfs:
        # Define the mapping for projections: (Summed Axis, Static Axis, Plane, Label)
        projections = [
            (0, "x", x.min(), "Y-Z"),
            (1, "y", y.min(), "X-Z"),
            (2, "z", z.min(), "X-Y")
        ]

        for axis_idx, static_dim, pos, label in projections:
            pdf_data = np.sum(values, axis=axis_idx)

            # Slice the 3D grids to get 2D planes for the other two dimensions
            # takes a slice at index 0 along the marginalized axis
            s = [slice(None)] * comb(len(projections), 2)
            s[axis_idx] = 0

            # Create surface kwargs
            surf_kwargs = {
                "x": x[tuple(s)],
                "y": y[tuple(s)],
                "z": z[tuple(s)],
                "surfacecolor": pdf_data,
                "colorscale": colorscale,
                "showscale": False,
                "opacity": 0.5,
                "name": f"PDF {label}",
                "hoverinfo": "skip",
            }

            # Lock the plane to the boundary wall
            surf_kwargs[static_dim] = np.full_like(surf_kwargs[static_dim], pos)

            fig.add_trace(go.Surface(**surf_kwargs))
