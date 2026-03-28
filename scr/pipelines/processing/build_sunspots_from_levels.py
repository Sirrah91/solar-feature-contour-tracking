from scr.utils.types_alias import AssociationMode, Sunspots, Metadata, Events, Quantity, SunspotPart
from scr.utils.dict import separate_key_value

from scr.io.fits.stack import LazyImageStack

from scr.sunspots.association import associate_levels_flat

from scr.tracks.tracking import track_contours
from scr.tracks.preprocessing import preprocess_tracks
from scr.tracks.labels import tracks_label
from scr.tracks.relabel import relabel_tracks_by_lifetime


def track_and_merge_sunspots(
        *,
        image_paths: list[str],
        contour_quantity: Quantity,
        components: list[dict[str, float | SunspotPart]],
        max_gap: int = 3,
        iou_threshold: float = 0.3,
        registration: bool = True,
        filtering: dict | None = None,
        containment_mode: AssociationMode = "covers",
        min_containment: float = 0.8,
) -> tuple[Sunspots, Metadata, Events]:
    """
    Track contours from an image sequence and build sunspots by
    associating outer and nested components (e.g. pores, umbrae).

    This pipeline performs:
        1. Contour detection
        2. Contour tracking
        3. Optional filtering of outer tracks
        4. Event pruning to remain consistent with filtering
        5. Sunspot association across components

    Parameters
    ----------
    image_paths : list[str]
        Paths of the evaluated images.

    contour_quantity : Quantity
        Physical quantity used to compute contours.

    components : list[dict[str, float | SunspotPart]
        Ordered list of component configurations.
        The first element is considered the outermost component.

        Example:
            [
                {"name": "penumbra", "level": 0.9,  "min_area": 3.0},
                {"name": "pore",     "level": 0.65, "min_area": 3.0},
                {"name": "umbra",    "level": 0.5,  "min_area": 3.0},
            ]

    max_gap : int
        Maximum number of consecutive missing frames allowed in tracking.

    iou_threshold : float
        Intersection-over-Union threshold used for track matching.

    registration : bool
        If True, register previous image to the current one
        before contour matching.

    filtering : dict, optional
        Optional filtering applied to outer tracks only.
        Supported keys:
            - min_frames : int
            - remove_nested : bool
            - collapse_nested : bool
            - min_vertices : int
            - max_healing_gap : float
            - max_closing_gap : float

        Filtering is followed by event pruning and relabelling.

    containment_mode : AssociationMode
        Mode passed to contour_belongs_to_outer.

    min_containment : float
        Minimum fraction of the smaller region required to be inside
        the larger one.

    Returns
    -------
    Sunspots
        Hierarchical sunspot structure.

    Metadata
        Processing metadata (frames, configuration, etc.).

    Events
        Track event log (splits, merges) consistent with final
        filtered and relabelled outer tracks.
    """
    # Determine if contours should be flipped for filled convention
    flip_contours = contour_quantity != "Ic"

    all_tracks = []
    outermost_events = None

    # Load images
    images = LazyImageStack(
        image_paths,
        contour_quantity,
    )

    # --------------------------------------------------
    # Track all levels
    # --------------------------------------------------

    for i, comp in enumerate(components):

        tracks, events = track_contours(
            images=images,
            level=comp["level"],
            min_area=comp["min_area"],
            max_gap=max_gap,
            iou_threshold=iou_threshold,
            registration=registration,
            flip_contours=flip_contours,
        )

        if i == 0:
            outermost_events = events

        all_tracks.append({
            comp.get("name", tracks_label(quantity=contour_quantity, level=comp["level"])): tracks
        })

    # --------------------------------------------------
    # Apply Option-1 filtering to OUTERMOST only
    # --------------------------------------------------

    outer_name, outer_tracks = separate_key_value(all_tracks[0])

    outer_tracks, events = preprocess_tracks(
        tracks=outer_tracks,
        filtering=filtering,
        events=outermost_events,
        min_containment=min_containment,
    )

    # --------------------------------------------------
    # Flat association
    # --------------------------------------------------

    outer_tracks, events = relabel_tracks_by_lifetime(
        tracks=outer_tracks,
        events=events
    )

    all_tracks[0] = {outer_name: outer_tracks}

    sunspots = associate_levels_flat(
        all_tracks=all_tracks,
        mode=containment_mode,
        min_fraction=min_containment,
    )

    metadata = {
        "image_paths": image_paths,
        "contour_quantity": contour_quantity,
        "components": components,
        "max_gap": max_gap,
        "iou_threshold": iou_threshold,
        "registration": registration,
        "filtering": filtering,
        "containment_mode": containment_mode,
        "min_containment": min_containment,
    }

    return sunspots, metadata, events
