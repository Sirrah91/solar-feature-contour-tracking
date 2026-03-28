from typing import Iterable, Mapping, Callable, Iterator
from matplotlib.axes import Axes

from scr.utils.types_alias import Quantity
from scr.plotting.types import ContourGroup
from scr.plotting.scene.frame_spec import FrameSpec
from scr.plotting.scene.frame_data import FrameData
from scr.plotting.builders.frames import frame_data_from_observation


def iter_frame_data(
        *,
        frame_specs: Iterable[FrameSpec],
        contour_source: Mapping,
        quantity: Quantity,
        contour_parser: Callable[..., list[ContourGroup]],
        style_resolver: Callable[..., dict] | None = None,
        annotations: Callable[[Axes, list[ContourGroup]], None] | None = None,
) -> Iterator[FrameData]:
    """
    Yield FrameData for each provided FrameSpec.
    """
    for spec in frame_specs:
        observation_id = spec.observation_id
        frame = spec.frame
        image_path = spec.image_path

        yield frame_data_from_observation(
            observation_id=observation_id,
            frame=frame,
            image_path=image_path,
            quantity=quantity,
            contour_source=contour_source,
            contour_parser=contour_parser,
            style_resolver=style_resolver,
            annotations=annotations,
        )
