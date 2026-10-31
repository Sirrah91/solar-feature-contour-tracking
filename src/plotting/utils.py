from typing import Any, Mapping


def merge_explicit_kwargs(
    base: Mapping[str, Any] | None,
    /,
    **explicit: Any,
) -> dict[str, Any]:
    """
    Merge explicit keyword arguments into a base kwargs mapping.

    Explicit values override base values *only if they are not None*.

    Parameters
    ----------
    base : Mapping or None
        Base keyword arguments (e.g. image_kwargs, hist_kwargs).
    **explicit
        Explicit keyword arguments with priority.

    Returns
    -------
    dict
        Merged kwargs dictionary.
    """
    merged = {} if base is None else dict(base)

    for key, value in explicit.items():
        if value is not None:
            merged[key] = value

    return merged
