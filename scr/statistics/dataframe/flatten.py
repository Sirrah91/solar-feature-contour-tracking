import numpy as np
import pandas as pd

from scr.config.numerics import WP
from scr.utils.types_alias import ObservationID, StatsByQuantity


def _is_topology_param(
        param: str
) -> bool:
    return param.startswith("topology")


def flatten_spot_features_with_frame(
        all_stats: dict[ObservationID, StatsByQuantity]
) -> pd.DataFrame:
    """
    Flatten nested sunspot statistics into a Pandas DataFrame.

    Output
    ------
    DataFrame where each row represents one (observation_id, sunspot_id, frame),
    dtype-optimised:
        - observation_id: category
        - sunspot_id: category
        - frame: category
        - spot_global_index: int32
        - topology_*: category
        - all other numeric values: float32
    """

    records = []

    for obs_id, quantities in all_stats.items():

        # All spot ids under any physical quantity
        spot_ids = set().union(*(q.keys() for q in quantities.values()))

        for spot_id in spot_ids:

            # Collect all frames for this spot
            all_frames = set()
            for spots in quantities.values():
                if spot_id not in spots:
                    continue

                spot_data = spots[spot_id]

                for part in spot_data:
                    all_frames.update(spots[spot_id][part].keys())

            # Process each frame
            for frame in all_frames:

                record = {
                    "observation_id": obs_id,
                    "sunspot_id": np.int32(spot_id),
                    "frame": np.int32(frame),
                }

                written_geom = set()  # track which regions already wrote non-flux params

                for phys_q, spots in quantities.items():
                    if spot_id not in spots:
                        continue

                    spot_data = spots[spot_id]

                    for part, part_data in spot_data.items():

                        if frame not in part_data:
                            continue

                        params = part_data[frame]

                        # (A) Flux parameters → quantity dependent
                        for param, val in params.items():
                            if "flux" in param:
                                key = f"{phys_q}_{part}_{param}"
                                record[key] = (
                                    WP(val if val is not None else np.nan)
                                )

                        # (B) Non-flux parameters → once per region per frame
                        if part not in written_geom:
                            for param, val in params.items():
                                if "flux" not in param:
                                    key = f"{part}_{param}"
                                    if _is_topology_param(param):
                                        record[key] = np.int32(val)
                                    else:
                                        record[key] = WP(val if val is not None else np.nan)

                            written_geom.add(part)

                records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Optimise ID columns
    df["observation_id"] = df["observation_id"].astype("category")
    df["sunspot_id"] = df["sunspot_id"].astype("category")
    df["frame"] = df["frame"].astype("category")

    # Topology columns → categorical
    topology_cols = [c for c in df.columns if _is_topology_param(c)]
    df[topology_cols] = df[topology_cols].astype("category")

    # ---- Sort here for stable output ----
    df.sort_values(["observation_id", "sunspot_id", "frame"], inplace=True)

    # ---- Add per-sunspot local index ----
    df["spot_global_index"] = (
        pd.factorize(
            pd.MultiIndex.from_frame(
                df[["observation_id", "sunspot_id"]]
            )
        )[0]
        .astype("int32")
    )

    # --------------------------------------------------
    # Topology columns → categorical
    # --------------------------------------------------
    topology_cols = [c for c in df.columns if c.startswith("topology")]

    for col in topology_cols:
        # convert from float32 to integer safely (NaNs preserved)
        df[col] = (
            df[col]
            .astype("Int16")   # nullable integer
            .astype("category")
        )

    df.reset_index(drop=True, inplace=True)

    return df
