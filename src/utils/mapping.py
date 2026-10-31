from typing import Any, Mapping

from src.utils.json import json_serial


def update_dict(
        base: dict[str, Any],
        updates: Mapping[str, Any],
        *,
        overwrite: bool = True,
) -> dict[str, Any]:
    """
    Recursively update *base* with *updates*.

    Parameters
    ----------
    base : dict
        Mapping to update **in-place**.
    updates : Mapping
        New key-value pairs.
    overwrite : bool, optional
        If False, existing keys in *base* are not replaced (deep-merge only
        for nested dicts).

    Returns
    -------
    dict
        The updated *base* (same object).
    """
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], Mapping)
            and isinstance(value, Mapping)
        ):
            update_dict(base[key], value, overwrite=overwrite)
        elif overwrite or key not in base:
            base[key] = value
    return base


def flatten_dict(
        d: dict[str, Any],
        *,
        parent_key: str = "",
        sep: str = "."
) -> dict[str, Any]:
    """
    Recursively flatten a nested dictionary, joining paths with a separator.

    Example: {'clouds': {'thresholds': {'min_contrast': 40.0}}}
    Becomes: {'clouds.thresholds.min_contrast': 40.0}
    """
    items = []
    for k, v in d.items():
        # Replace spaces in the key fragments early
        clean_k = k.replace(" ", "_")
        new_key = f"{parent_key}{sep}{clean_k}" if parent_key else clean_k

        # 1. Handle recursion for dictionaries
        if isinstance(v, dict):
            items.extend(flatten_dict(v, parent_key=new_key, sep=sep).items())
            continue

        # 2. Leverage your json_serial to handle complex objects
        try:
            serialised_v = json_serial(v)
        except TypeError:
            # Fallback for types json_serial doesn't know about
            serialised_v = v

        # 3. Handle specific formatting for FITS arrays/units
        if isinstance(serialised_v, list | tuple):
            # If json_serial converted a numpy array or list, stringify it safely for FITS
            str_elements = [str(item) for item in serialised_v]
            items.append((new_key, ", ".join(str_elements)))

        else:
            # Everything else (int, float, str, bool) passes through as its native primitive type
            items.append((new_key, serialised_v))

    return dict(items)
