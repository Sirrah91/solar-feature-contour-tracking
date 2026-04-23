import numpy as np
import pandas as pd
from typing import Sized

from scr.utils.types_alias import Stat


def _flatten_scalar_stat(
        prefix: str,
        value,
        flat: dict
) -> None:
    """
    Flatten scalar, tuple, or dict global statistic.
    """

    # Dict case (e.g. mu, flux)
    if isinstance(value, dict) and len(value) > 0:
        for k, v in value.items():
            flat[f"{prefix}_{k}"] = v
        return

    # Scalar
    flat[prefix] = value


def _flatten_per_object(
        prefix: str,
        per_obj,
        flat: dict
) -> None:
    """
    Flatten per-object statistics.
    Supports:
        - list of scalars
        - list of tuples
        - dict of lists
    """

    if not isinstance(per_obj, Sized) or len(per_obj) == 0:
        flat[f"{prefix}_list"] = per_obj
        return

    # --- Dict of lists case (mu, flux after conversion) ---
    if isinstance(per_obj, dict) and len(per_obj) > 0:
        for k, v in per_obj.items():
            flat[f"{prefix}_{k}_list"] = v
        return

    # Scalar list
    flat[f"{prefix}_list"] = per_obj


def _flatten_stat_entry(
        prefix: str,
        value,
        flat: dict
) -> None:

    if isinstance(value, dict):

        if "global" in value:
            _flatten_scalar_stat(prefix, value["global"], flat)

        if "per_object" in value:
            _flatten_per_object(prefix, value["per_object"], flat)

        if "n_components" in value:
            flat[f"{prefix}-components"] = value["n_components"]
            flat[f"{prefix}-holes"] = value["n_holes"]

    else:
        _flatten_scalar_stat(prefix, value, flat)


def flatten_region_stats(
        mask_stats: dict,
        contour_stats: dict
) -> Stat:
    flat = {}

    for key, value in mask_stats.items():
        _flatten_stat_entry(key, value, flat)

    for key, value in contour_stats.items():
        _flatten_stat_entry(key, value, flat)

    return flat


def phase_duration_statistics(
        df: pd.DataFrame,
        value_col: str,
        duration_col: str = "phase_duration",
        percentiles=(95, 98),
) -> Stat:
    """
    Aggregate per (observation, sunspot, phase segment).
    """
    groups = df.groupby(
        ["spot_global_index", duration_col],
        observed=True,
    )

    duration = []
    max_val = []
    perc_vals = {p: [] for p in percentiles}

    for _, g in groups:
        duration.append(np.nanmax(g[duration_col]))
        vals = np.abs(g[value_col])

        max_val.append(np.nanmax(vals))
        for p in percentiles:
            perc_vals[p].append(
                np.nanpercentile(vals, p, method="median_unbiased")
            )

    out = {
        "duration": np.asarray(duration),
        "max": np.asarray(max_val),
    }
    for p in percentiles:
        out[f"p{p}"] = np.asarray(perc_vals[p])

    return out


def lifetime_and_mean(
        df: pd.DataFrame,
        value_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lifetime = number of valid frames.
    """
    groups = df.groupby(["spot_global_index"], observed=True)

    lifetime = []
    mean_val = []

    for _, g in groups:
        mask = np.isfinite(g[value_col])
        lifetime.append(mask.sum())
        mean_val.append(np.nanmean(g.loc[mask, value_col]))

    return np.asarray(lifetime), np.asarray(mean_val)
