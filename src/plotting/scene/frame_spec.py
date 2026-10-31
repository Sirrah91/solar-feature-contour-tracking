from typing import NamedTuple

from src.utils.types_alias import ObservationID, FrameID


class FrameSpec(NamedTuple):
    observation_id: ObservationID
    frame: FrameID
    image_path: str
