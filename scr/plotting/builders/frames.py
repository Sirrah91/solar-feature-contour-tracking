from typing import Callable, Mapping
from matplotlib.axes import Axes

from scr.utils.types_alias import Quantity
from scr.io.fits.read import load_image

from scr.plotting.types import ContourGroup
from scr.plotting.scene.frame_data import FrameData
from scr.plotting.style.resolvers import default_style_resolver


def frame_data_from_observation(
        *,
        observation_id: str,
        frame: int,
        image_path: str,
        quantity: Quantity,
        contour_source: Mapping,
        contour_parser: Callable[..., list[ContourGroup]],
        style_resolver: Callable[..., dict] | None = None,
        image_kwargs: dict | None = None,
        contour_kwargs: dict | None = None,
        annotations: Callable[[Axes, list[ContourGroup]], None] | None = None,
) -> FrameData:
    """
    Build FrameData for a single observation/frame using an injected contour parser.
    """
    image = load_image(image_path, quantity=quantity)

    if style_resolver is None:
        style_resolver = default_style_resolver()

    contour_groups = contour_parser(
        nested_tracks=contour_source,
        observation_id=observation_id,
        frame=frame,
        style_resolver=style_resolver,
    )

    return FrameData(
        image=image,
        contour_groups=contour_groups,
        image_kwargs=image_kwargs or {},
        contour_kwargs=contour_kwargs or {},
        annotations=annotations,
    )
