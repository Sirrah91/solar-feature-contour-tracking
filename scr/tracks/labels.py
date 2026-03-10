from scr.utils.types_alias import Quantity


def tracks_label(
        quantity: Quantity,
        level: float | int
) -> str:
    """
    Build a consistent label for tracks from its quantity and level.
    Directional symbol indicates “filled above/below”:
        '<' for Ic (bright above level)
        '>' for B  (filled below level)

    Example:
        >>> tracks_label("Ic", 0.5)
        'Ic<0.5'
        >>> tracks_label("B", 100)
        'B>100'
    """
    relation = "<" if quantity == "Ic" else ">"
    return f"{quantity}{relation}{level}"
