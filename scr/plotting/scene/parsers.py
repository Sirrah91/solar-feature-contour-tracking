from typing import Callable

from scr.plotting.types import ContourGroup
from scr.utils.types_alias import Tracks, Sunspots, SunspotsPhases, ObservationID


def contour_groups_from_tracks(
        *,
        nested_tracks: dict[ObservationID, Tracks],
        observation_id: ObservationID,
        frame: int,
        style_resolver: Callable[[int], dict],
) -> list[ContourGroup]:
    """
    Parse Tracks structure into ContourGroups for one frame.

    Expected structure:
        tracks[obs_id][track_id][frame] -> list[contours]
    """
    contour_groups: list[ContourGroup] = []

    obs_tracks = nested_tracks.get(observation_id, {})

    for track_id, track in obs_tracks.items():
        contours = track.get(frame)
        if not contours:
            continue

        contour_groups.append(
            ContourGroup(
                contours=contours,
                style=style_resolver(track_id),
                label=str(track_id),
            )
        )

    return contour_groups


def contour_groups_from_sunspots(
        *,
        nested_tracks: dict[ObservationID, Sunspots],
        observation_id: str,
        frame: int,
        style_resolver: Callable[[int], dict],
) -> list[ContourGroup]:
    """
    Parse Sunspots structure (no phases) into ContourGroups for one frame.

    Expected structure:
        sunspots[obs_id][sunspot_id][part][frame] -> list[contours]
    """
    contour_groups: list[ContourGroup] = []

    obs_data = nested_tracks.get(observation_id, {})

    for sunspot_id, sunspot_data in obs_data.items():
        merged: list = []

        for part_tracks in sunspot_data.values():
            merged.extend(part_tracks.get(frame, []))

        if not merged:
            continue

        contour_groups.append(
            ContourGroup(
                contours=merged,
                style=style_resolver(sunspot_id),
                label=str(sunspot_id),
            )
        )

    return contour_groups


def contour_groups_from_sunspot_phases(
        *,
        nested_tracks: dict[ObservationID, SunspotsPhases],
        observation_id: str,
        frame: int,
        style_resolver: Callable[[int, str], dict],
) -> list[ContourGroup]:
    """
    Parse SunspotsPhases structure into ContourGroups for one frame.

    Expected structure:
        sunspots_phases[obs_id][sunspot_id][phase][part][frame] -> list[contours]
    """
    contour_groups: list[ContourGroup] = []

    obs_data = nested_tracks.get(observation_id, {})

    for sunspot_id, sunspot_data in obs_data.items():

        # exactly one phase per sunspot per frame
        for phase, phase_data in sunspot_data.items():

            merged: list = []

            for part_tracks in phase_data.values():
                merged.extend(part_tracks.get(frame, []))

            if not merged:
                continue

            contour_groups.append(
                ContourGroup(
                    contours=merged,
                    style=style_resolver(sunspot_id, phase),
                    label=str(sunspot_id),
                    # label=f"{sunspot_id}:{phase}",
                )
            )

    return contour_groups
