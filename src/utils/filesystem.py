import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Sized

from src.utils.logging import get_logger

logger = get_logger("utils.filesystem")


def mkdir(
        dir_or_file_path: str,
        *,
        is_file_path: bool = False
) -> None:
    """
    Ensure that a directory exists.

    If `is_file_path` is True, create the parent directory of the given file path.
    If `is_file_path` is False, treat the input as a directory path.

    No permissions are changed here.
    """
    path = parse_path(dir_or_file_path)
    directory = path.parent if is_file_path else path

    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {directory}")

    created = not directory.exists()
    directory.mkdir(parents=True, exist_ok=True)
    if created:
        logger.info(f"Created directory: {directory}")


def parse_path(path_object: Path | str | None) -> Path | None:
    """
    Normalise a path value to :class:`~pathlib.Path` or ``None``.

    Mirrors :func:`~src.time.parsing.parse_datetime`  accepts whatever
    the caller has (string, Path, or None) and returns a canonical type.

    Parameters
    ----------
    path_object : str, Path, or None
        Input path.

    Returns
    -------
    Path or None
        ``None`` if *path* is ``None`` or an empty string.

    Examples
    --------
    >>> parse_path("/data/fits")
    PosixPath('/data/fits')
    >>> parse_path(Path("/data/fits"))
    PosixPath('/data/fits')
    >>> parse_path(None) is None
    True
    >>> parse_path("") is None
    True
    """
    if path_object is None:
        return None
    if isinstance(path_object, Path):
        return path_object.expanduser()
    stripped = str(path_object).strip()
    return Path(stripped).expanduser() if stripped else None


def is_empty(
        data
) -> bool:
    if data is None:
        return True

    if isinstance(data, np.ndarray):  # len(np.array([[]]))) = 1, use np.size()
        return data.size == 0

    if isinstance(data, pd.DataFrame):
        return data.empty

    if isinstance(data, Sized):  # those who have len()
        return len(data) == 0

    return False


def remove_if_exists(filename: Path | str | None) -> None:
    filename = str(filename)
    if filename is not None and os.path.isfile(filename):
        os.remove(filename)
