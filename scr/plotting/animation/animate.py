from matplotlib.artist import Artist
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable, Iterable, Iterator

from scr.plotting.scene.frame_data import FrameData
from scr.plotting.scene.render import render_scene


def animate_frames(
        frames: Iterable[FrameData],
        draw_frame: Callable[[Axes, FrameData], Iterable[Artist]],
        save_path: str,
        *,
        interval: int = 200,
        dpi: int = 100,
        figsize: tuple[float, float] = (8, 8),
        animation_kwargs: dict | None = None,
) -> FuncAnimation:
    """
    Animate a sequence or generator of FrameData using draw_frame.
    """
    if animation_kwargs is None:
        animation_kwargs = {}

    fig, ax = plt.subplots(figsize=figsize)

    def update(frame: FrameData) -> Iterable[Artist]:
        ax.clear()
        draw_frame(ax, frame)
        return ax.artists

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        cache_frame_data=False,
        **animation_kwargs,
    )

    ani.save(save_path, dpi=dpi, writer="ffmpeg")
    plt.close(fig)

    return ani


def animate_frames_from_generator(
        *,
        frames: Iterator[FrameData],
        save_path: str,
        interval: int = 200,
        dpi: int = 100,
        figsize: tuple[float, float] = (8, 8),
        animation_kwargs: dict | None = None,
) -> FuncAnimation:
    """
    Animate frames provided as a generator or iterable of FrameData.
    """
    def draw_frame(ax: Axes, frame: FrameData) -> Iterable[Artist]:
        ax.clear()
        render_scene(
            ax,
            image=frame.image,
            contour_groups=frame.contour_groups,
            image_kwargs=frame.image_kwargs,
            contour_kwargs=frame.contour_kwargs,
            annotations=frame.annotations,
        )
        ax.axis("off")
        return ax.artists

    return animate_frames(
        frames=frames,
        draw_frame=draw_frame,
        save_path=save_path,
        interval=interval,
        dpi=dpi,
        figsize=figsize,
        animation_kwargs=animation_kwargs,
    )
