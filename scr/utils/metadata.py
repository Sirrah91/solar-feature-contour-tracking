from typing import Any, Iterable


def merge_metadata(
        left: dict[Any, Any],
        right: dict[Any, Any],
        placeholders: Any | Iterable[Any] = "",
) -> dict[Any, Any]:
    """
    Merges two metadata dictionaries with placeholder awareness.

    The values in the `right` dictionary take priority unless the value is
    found in `placeholders`, in which case the value from `left` is preserved.

    Parameters
    ----------
    left : dict
        The base dictionary (typically current command-line arguments).
    right : dict
        The priority dictionary (typically historical metadata from a file).
    placeholders : Any or Iterable of Any, optional
        Values to be treated as empty placeholders. If the `right` dictionary
        contains one of these values, the `left` value is kept.
        Default is "" (empty string).

    Returns
    -------
    dict
        A new dictionary containing the merged metadata.
    """
    # Ensure placeholders is a collection for consistent 'in' checks
    if isinstance(placeholders, (str, bytes)) or not isinstance(placeholders, Iterable):
        placeholder_list = [placeholders]
    else:
        placeholder_list = placeholders

    # Start with a copy of the left (CLI args / Base)
    merged = left.copy()

    # Iterate through the right (Historical / Priority)
    for key, val in right.items():
        # Only overwrite if the right value is NOT a placeholder
        if val not in placeholder_list:
            merged[key] = val

    return merged
