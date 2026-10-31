import astropy.units as u


def plural(
        n: int,
        word: str,
        *,
        suffix: str = "s"
) -> str:
    """Returns a string with the number and the pluralized word if n != 1."""
    return f"{n} {word}" + (suffix if n != 1 else "")


def parse_unit(unit: u.Quantity) -> str:
    return f"{unit.value} {unit.unit}"
