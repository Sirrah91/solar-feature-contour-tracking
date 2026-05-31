"""
Filename conventions for convective regime data files.

All save and load sites import from here so a naming change
only needs to happen in one place.
"""
from os import path
from scr.convective_regimes.utils.types_alias import FilterMode, SunspotPhase, Quantity


def extract_filename(
        *,
        data_dir: str,
        object_type: FilterMode,
        phase: SunspotPhase,
        suf: str = ""
) -> str:
    return path.join(
        data_dir,
        f"{object_type}_{phase}{suf}.npz",
    )


def probability_filename(
        *,
        data_dir: str,
        object_type: FilterMode,
        phase: SunspotPhase,
        region: str,
        quantity: Quantity,
        suf: str = ""
) -> str:
    return path.join(
        data_dir,
        f"{object_type}_{phase}_{region}_{quantity}_probability{suf}.npz",
    )


def regression_filename(
        *,
        data_dir: str,
        object_type: FilterMode,
        phase: SunspotPhase,
        region: str,
        quantity: Quantity = "",
        suf: str = ""
) -> str:
    return path.join(
        data_dir,
        f"{object_type}_{phase}_{region}_{quantity}_regression{suf}.parquet",
    )
