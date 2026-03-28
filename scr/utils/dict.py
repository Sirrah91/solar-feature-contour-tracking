from typing import Mapping


def separate_key_value(
        dictionary: dict,
) -> tuple:
    return next(iter(dictionary.items()))


def normalize_dicts(obj):
    if isinstance(obj, Mapping):
        return {k: normalize_dicts(v) for k, v in obj.items()}
    return obj
