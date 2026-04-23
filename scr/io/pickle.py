import pickle
from scr.utils.dict import normalize_dicts


def load_pickle(filename: str):
    """Load a pickle file without schema assumptions."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(filename: str, obj) -> None:
    """Save a pickle file without schema assumptions."""
    obj = normalize_dicts(obj)
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
