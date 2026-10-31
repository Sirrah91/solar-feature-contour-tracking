"""JSON Serialisation Utilities."""
from datetime import datetime, date
from pathlib import Path
from typing import Any

import numpy as np

try:
    from astropy.time import Time
    import astropy.units as u
    _ASTROPY = True
except ImportError:
    _ASTROPY = False

from src.utils.strings import parse_unit


def json_serial(obj: Any) -> Any:
    """
    Extended JSON serialiser for types not handled by the default encoder.

    Pass as the *default* argument to ``json.dumps`` / ``json.dump``::

        json.dumps(my_dict, default=json_serial, indent=4)

    Handles: ``datetime``, ``date``, astropy ``Time`` and ``Quantity``,
    numpy scalars/arrays, ``Path``, ``bytes``.

    Raises
    ------
    TypeError
        For any type not covered above.
    """
    # 1. Check Astropy types FIRST (because Quantity is a subclass of np.ndarray)
    if _ASTROPY:
        if isinstance(obj, Time):
            return obj.isot
        if isinstance(obj, u.Quantity) or (hasattr(obj, "value") and hasattr(obj, "unit")):
            return parse_unit(obj)

    # 2. Standard library datetime types
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8")

    # 3. Generic NumPy types (only hits if it survived the Quantity check)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(
        f"Object of type {type(obj).__name__!r} is not JSON serialisable."
    )
