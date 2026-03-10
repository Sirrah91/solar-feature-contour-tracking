from scr.utils.types_alias import TracksLabeled, Events, Quantity, Metadata
from scr.io.fits.stack import LazyImageStack
from scr.tracks.tracking import track_contours as _track_contours
from scr.tracks.labels import tracks_label


def track_contours(
        image_paths: list[str],
        contour_level: float,
        contour_quantity: Quantity,
        min_area: float = 5.,
        max_gap: int = 3,
        iou_threshold: float = 0.3,
        registration: bool = True,
) -> tuple[TracksLabeled, Metadata, Events]:
    """
    Detect and track contours across an image sequence.

    Parameters
    ----------
    image_paths : list[str]
        Paths of the evaluated images.

    contour_level : float
        Contour threshold level.

    contour_quantity : Quantity
        Physical quantity used to compute contours.

    min_area : float
        Minimum contour area threshold (pixels).

    max_gap : int
        Maximum number of consecutive missing frames allowed in tracking.

    iou_threshold : float
        Intersection-over-Union threshold for track matching.

    registration : bool
        If True, register previous image to the current one
        before contour matching.

    Returns
    -------
    TracksLabeled
        Dictionary mapping track_id → {frame_index: contour}.

    Events
        List of tracking events (splits and merges) describing
        track lineage relationships.
    """
    # Determine if contours should be flipped for filled convention
    flip_contours = contour_quantity != "Ic"

    # Load images
    images = LazyImageStack(
        image_paths,
        contour_quantity,
    )

    # Track contours
    tracks, events = _track_contours(
        images=images,
        level=contour_level,
        min_area=min_area,
        max_gap=max_gap,
        iou_threshold=iou_threshold,
        registration=registration,
        flip_contours=flip_contours,
    )

    label = tracks_label(quantity=contour_quantity, level=contour_level)

    metadata = {
        "image_paths": image_paths,
        "contour_quantity": contour_quantity,
        "contour_level": contour_level,
        "min_area": min_area,
        "max_gap": max_gap,
        "iou_threshold": iou_threshold,
        "registration": registration
    }

    return {label: tracks}, metadata, events
