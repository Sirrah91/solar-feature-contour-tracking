from scr.utils.types_alias import Sunspots, SunspotPart, Quantity, Events, Metadata, AssociationMode
from scr.utils.dict import separate_key_value

from scr.io.fits.stack import LazyImageStack
from scr.io.tracks import load_track_file
from scr.sunspots.builder import build_sunspots_from_outer_tracks

from scr.tracks.preprocessing import preprocess_tracks
from scr.tracks.labels import tracks_label

"""
track_path = "/nfsscratch/david/Contours/tracks/tracks_Ic-0.9_AR-11108_20100920_S30E36.npz"
quantity = "Ic"
containment_mode = "covers"
min_containment = 0.8
filtering = {'min_frames': 3,
  'collapse_nested': True,
  'remove_nested': False,
  'min_vertices': 4,
  'max_healing_gap': 0.0,
  'max_closing_gap': 0.0}
components = [
    {'name': 'Ic<0.65', 'level': 0.65, 'min_area': 3.0},
    {'name': 'Ic<0.5', 'level': 0.5, 'min_area': 3.0},
]
"""

def run_sunspot_association_pipeline(
        *,
        track_path: str,
        quantity: Quantity,
        components: list[dict[str, float | SunspotPart]],
        filtering: dict | None = None,
        containment_mode: AssociationMode = "covers",
        min_containment: float = 0.8,
) -> tuple[Sunspots, Metadata, Events]:
    """
    Build sunspots from precomputed contour tracks by analysing
    containment relations between outer tracks and nested components.

    No contour detection or tracking is performed.

    The pipeline performs:
        1. Loading of stored contour tracks
        2. Optional filtering of outer tracks
        3. Event pruning to remain consistent with filtering
        4. Sunspot association

    Parameters
    ----------
    track_path : str
        Path to stored contour tracks file.

    quantity : Quantity
        Physical quantity from which contours were computed.

    components : list[dict[str, float | SunspotPart]
        Ordered list of component configurations.
        The first element is considered the outermost component.

    components : list[dict[str, float | SunspotPart]
        List of nested component configurations.

        Example:
            [
                {"name": "penumbra", "level": 0.9,  "min_area": 3.0},
                {"name": "pore",     "level": 0.65, "min_area": 3.0},
                {"name": "umbra",    "level": 0.5,  "min_area": 3.0},
            ]

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
        Passed to contour_belongs_to_outer.

    min_containment : float
        Minimum fraction of smaller region inside larger one.

    Returns
    -------
    Sunspots
        Hierarchical sunspot structure.

    Metadata
        Processing metadata.

    Events
        Track event log consistent with final filtered
        and relabelled outer tracks.
    """

    # ------------------------------------------------------------------
    # Load tracks and images
    # ------------------------------------------------------------------
    tracks_dict, _, metadata, events = load_track_file(track_path)

    if len(tracks_dict) != 1:
        raise ValueError(
            f"Expected exactly one outer track set, got {len(tracks_dict)}."
        )

    outer_name, outer_tracks = separate_key_value(tracks_dict)

    # Load images
    images = LazyImageStack(
        metadata["image_paths"],
        quantity,
    )

    # ------------------------------------------------------------------
    # Optional filtering
    # ------------------------------------------------------------------
    outer_tracks, events = preprocess_tracks(
        tracks=outer_tracks,
        filtering=filtering,
        events=events,
        min_containment=min_containment,
    )

    # ------------------------------------------------------------------
    # Prepare component configuration
    # ------------------------------------------------------------------
    levels = {
        component.get("name", tracks_label(quantity=quantity, level=component["level"])): component["level"]
        for component in components
    }
    min_area = {
        component.get("name", tracks_label(quantity=quantity, level=component["level"])): component.get("min_area", 0.0)
        for component in components
    }

    # ------------------------------------------------------------------
    # Build sunspots
    # ------------------------------------------------------------------
    sunspots = build_sunspots_from_outer_tracks(
        images=images,
        outer_tracks={outer_name: outer_tracks},
        levels=levels,
        min_area=min_area,
        containment_mode=containment_mode,
        min_containment=min_containment,
    )

    # ------------------------------------------------------------------
    # Collect metadata
    # ------------------------------------------------------------------
    # Add the outer component for consistency
    components.insert(
        0,
        {"name": outer_name, "level": metadata["contour_level"], "min_area": metadata["min_area"]},
    )

    metadata |= {
        "track_path": track_path,
        "inner_contour_quantity": quantity,
        "components": components,
        "filtering": filtering,
        "containment_mode": containment_mode,
        "min_containment": min_containment,
    }

    return sunspots, metadata, events
