import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sized


def check_dir(
        dir_or_file_path: str,
        is_file: bool = False
) -> None:
    """
    This function checks if the directory exists. If not, the function creates it.
    """

    dir_or_file_path = Path(dir_or_file_path)

    if is_file:
        directory = dir_or_file_path.parent
    else:
        directory = dir_or_file_path

    if not directory.is_dir():
        print(f'Directory "{directory.as_posix()}" does not exist, creating it now.')
        directory.mkdir(parents=True, exist_ok=True)


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
