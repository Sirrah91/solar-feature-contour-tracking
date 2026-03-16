import pandas as pd

from scr.utils.types_alias import ObservationID, Sunspots, SunspotsPhasesByObservation
from scr.utils.collections import nested_defaultdict


def split_by_phase(
        combined_df: pd.DataFrame,
        all_sunspots: dict[ObservationID, Sunspots],
) -> SunspotsPhasesByObservation:
    sunspots_phases = nested_defaultdict(factory=list, depth=5)

    for _, row in combined_df.iterrows():
        ph = row.phase
        if ph.lower() not in ["forming", "stable", "decaying"]:
            continue
        fid, sid, frame = row.observation_id, row.sunspot_id, row.frame

        # Copy contours
        spot = all_sunspots[fid].get(sid, {})
        for region in spot:
            if frame in spot.get(region, {}):
                sunspots_phases[fid][sid][ph][region][frame].extend(spot[region][frame])

    return sunspots_phases
