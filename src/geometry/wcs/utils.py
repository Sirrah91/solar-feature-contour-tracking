"""Header Update Utilities."""
from typing import Any

import numpy as np

from src.utils.types_alias import Header
from src.utils.mapping import flatten_dict


class HeaderReconstructionError(KeyError):
    """Raised when a header is missing something we refuse to guess at."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_header_value(value: Any) -> Any:
    """
    Convert a scalar value to a format that Astropy safely accepts as a FITS keyword.

    - ``None``, ``bool``                        → ``str(value)``
    - Non-finite ``float`` / ``int`` (NaN, Inf)  → ``str(value)``
    - Anything else                             → returned unchanged
    """
    if value is None:
        return str(value)

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (float, int, np.floating, np.integer)):
        if not np.isfinite(value):
            return str(value)

    return value


# ---------------------------------------------------------------------------
# Header updates
# ---------------------------------------------------------------------------

def reset_rotation(header: Header) -> None:
    """
    Replace all rotation-related WCS keywords with the identity transform.
    """
    header["CROTA2"] = (0.0, "[deg] Rotation angle (reset to 0)")
    header["ANGLE"] = (0.0, "[deg] Rotation angle (= CROTA2) (reset to 0)")

    header["SOLAR_P0"] = (0.0, "[deg] Solar P angle (reset to 0)")

    header["PC1_1"] = (1.0, "Identity transform PC1_1")
    header["PC1_2"] = (0.0, "Identity transform PC1_2")
    header["PC2_1"] = (0.0, "Identity transform PC2_1")
    header["PC2_2"] = (1.0, "Identity transform PC2_2")


def alias_keyword(
        header: Header,
        *,
        source: str,
        target: str,
        comment: str | None = None,
) -> None:
    """
    One-way FITS keyword alias.
    """

    if target in header:
        return
    if source not in header:
        return

    header[target] = (
        header[source],
        comment or f"Alias of {source}",
    )


def update_fits_header(
        header: Header,
        header_update_json: dict | None,
) -> Header:
    """
    Apply a dictionary of keyword updates to *header*.

    Values may be raw scalars or ``{"value": ..., "comment": ...}`` dicts.
    """
    if not header_update_json:
        return header
    for key, entry in header_update_json.items():
        if isinstance(entry, dict) and "value" in entry:
            header.set(key, entry["value"], entry.get("comment", ""))
        else:
            header[key] = entry
    return header


def add_sol_standard_to_header(
        header: Header,
        sol_list: list[str],
) -> Header:
    """Add SOL-standard event identifiers to *header*."""
    header.set("SOLCOUNT", len(sol_list), "Number of SOL standard entries")
    for i, sol in enumerate(sol_list):
        key = f"SOL{i}"
        if len(key) <= 8:
            header.set(key, sol, "SOL standard")
        else:
            header[f"HIERARCH {key}"] = sol
    return header


def create_args_header(args_dict: dict) -> Header:
    """
    Build a FITS header from an arbitrary nested key-value dict using HIERARCH keys.
    """
    header = Header()

    # 1. Flatten the configuration hierarchy and compress extreme keys
    flat_args = clean_config_for_fits(flatten_dict(args_dict, sep="."))

    # 2. Iterate and append safely
    for key, value in flat_args.items():
        # Clean collection sequences into safe flat strings (e.g., ('FL',) -> 'FL')
        if isinstance(value, (list, tuple, set)):
            value = ", ".join(str(item) for item in value)

        try:
            header[f"HIERARCH {key}"] = _safe_header_value(value)
        except Exception as exc:
            header.add_comment(f"{key} = {value} (fallback: {exc})")

    return header


def clean_config_for_fits(flat_config: dict) -> dict:
    """Shortens extreme nested dot-notation keys to prevent FITS CONTINUE bugs."""
    fits_ready = {}
    for key, value in flat_config.items():
        # Strip long repeating prefixes to save card space
        short_key = key
        if short_key.startswith("observation.instrument.cloud_thresholds_override."):
            short_key = short_key.replace("observation.instrument.cloud_thresholds_override.", "INST.CLD_THR.")
        elif short_key.startswith("observation.instrument.wavelength.default_cloud_thresholds."):
            short_key = short_key.replace("observation.instrument.wavelength.default_cloud_thresholds.", "WAVE.CLD_THR.")
        elif short_key.startswith("observation.instrument.wavelength."):
            short_key = short_key.replace("observation.instrument.wavelength.", "WAVE.")
        elif short_key.startswith("observation.instrument."):
            short_key = short_key.replace("observation.instrument.", "INST.")
        elif short_key.startswith("observation.observatory."):
            short_key = short_key.replace("observation.observatory.", "OBS.")

        # FITS struggles parsing float('inf') or float('nan') into standard cards
        # Convert them explicitly to clean strings or fixed placeholders
        import math
        if isinstance(value, float):
            if math.isinf(value):
                value = "INF"
            elif math.isnan(value):
                value = "NAN"

        fits_ready[short_key] = value
    return fits_ready


# ---------------------------------------------------------------------------
# Values rounding
# ---------------------------------------------------------------------------

def safe_round(number: float | None, prec: int | None = 6) -> float | None:
    if number is None:
        return None
    return round(number, prec) if prec is not None else number
