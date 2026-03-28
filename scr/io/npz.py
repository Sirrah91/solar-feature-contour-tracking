import numpy as np
from numpy.lib.npyio import NpzFile
from scr.utils.dict import normalize_dicts


def load_npz(filename: str) -> NpzFile:
    """Load a NumPy .npz archive without interpreting contents."""
    return np.load(filename, allow_pickle=True)


def save_npz(filename: str, **kwargs) -> None:
    """Save a NumPy .npz archive without interpreting contents."""
    clean = {
        k: normalize_dicts(v) if not isinstance(v, np.ndarray) else v
        for k, v in kwargs.items()
    }
    np.savez_compressed(filename, **clean, allow_pickle=True)
