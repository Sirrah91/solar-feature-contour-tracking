import numpy as np

from scr.io.npz import load_npz
from scr.convective_regimes.core.models import ProbabilityMap2D
from scr.convective_regimes.io.filenames import extract_filename
from scr.convective_regimes.settings import DATA_DIR
from scr.convective_regimes.utils.types_alias import FilterMode, SunspotPhase, Quantity, Boundary


_PHASES_ALL: tuple[SunspotPhase, ...] = ("forming", "stable", "decaying")


def load_all(
        filter_modes: list[FilterMode],
        phases: list[SunspotPhase] = _PHASES_ALL,
        *,
        data_dir: str = DATA_DIR,
) -> list:
    """
    Load npz data files for the given filter modes and phases.

    Passing phases=("all",) expands to all three canonical phases.

    Parameters
    ----------
    filter_modes : e.g. ("sunspots",) or ("pores", "sunspots")
    phases : one or more of "forming", "stable", "decaying", or ("all",)
    data_dir : root directory containing the npz files
    """
    if len(phases) == 1 and phases[0] == "all":
        phases = _PHASES_ALL

    data_all = []
    for filter_mode in filter_modes:
        for phase in phases:
            filename = extract_filename(
                data_dir=data_dir,
                object_type=filter_mode,
                phase=phase,
            )
            data_all.append(load_npz(filename))
    return data_all


def extract_q(
        data_all: list,
        q: Quantity,
        per_object: bool = False,
        boundary: Boundary = "b605.0",
) -> np.ndarray:
    """
    Extract and concatenate a quantity across all loaded data files.

    Parameters
    ----------
    data_all : list of npz archives from load_all
    q : quantity key, e.g. "Br", "Bhor", "Binc", "Ic", "Phi"
    per_object : if True return a list of per-object arrays (dtype=object);
                 if False flatten all pixels into a single float array
    boundary : region suffix used in the npz key, e.g. "b605.0", "ic0.9"

    Returns
    -------
    np.ndarray of dtype object (per_object=True) or float (per_object=False)
    """
    y = []
    for data in data_all:
        values = list(data[f"{q}_{boundary}"])
        if q == "Binc":
            values = [np.rad2deg(v) for v in values]
        else:
            values = [np.abs(v) for v in values]

        if per_object:
            y.extend(values)
        else:
            y.extend(np.concatenate(values))

    if per_object:
        return np.asarray(y, dtype=object)
    return np.asarray(y, dtype=float)


def load_probability_map(filepath: str) -> ProbabilityMap2D:
    """
    Load a saved probability-map npz and return a typed ProbabilityMap2D.

    Handles the legacy dict-in-npz storage format written by the compute
    script. For files written after the refactor, prefer loading the fields
    directly.
    """
    data = load_npz(filepath)

    return ProbabilityMap2D(
        probability=data["probability"],
        counts=data["counts"],
        x_bins=data["x_bins"],
        y_bins=data["y_bins"],
    )
