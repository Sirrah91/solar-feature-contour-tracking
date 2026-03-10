from os import path
from scr.config.naming import SEP_OUT, SEP_IN
from scr.utils.types_alias import Quantity


def plural(
        n: int,
        word: str,
        suffix: str = "s"
) -> str:
    """Returns a string with the number and the pluralized word if n != 1."""
    return f"{n} {word}" + (suffix if n != 1 else "")


def ensure_extension(
        filename: str,
        extension: str,
) -> str:
    if not filename.endswith(extension):
        filename += extension
    return filename


def get_initial_track_filename(
        data_dir: str,
        quantity: str,
        level: float
) -> str:
    """
    Creates the standardized initial track filename.
    Example: 'tracks_Ic<0.9_HARP-00970_20111021.npz'
    """
    # normpath + basename handles trailing slashes correctly
    dir_name = path.basename(path.normpath(data_dir))

    track_filename = (
        f"tracks{SEP_OUT}"
        f"{quantity}{SEP_IN}{level}{SEP_OUT}"
        f"{dir_name}.npz"
    )

    return ensure_extension(track_filename, extension=".npz")


def track_to_sunspot_filename(
        track_path: str
) -> str:
    """
    Converts a track filename to a sunspot filename.
    Example: 'tracks_Ic<0.9.npz' -> 'sunspots_Ic<0.9.npz'
    """
    base = path.basename(track_path)
    prefix = f"tracks{SEP_OUT}"
    new_prefix = f"sunspots{SEP_OUT}"

    if base.startswith(prefix):
        # Swap the prefix
        sunspot_filename = base.replace(prefix, new_prefix, 1)
    else:
        # Prepend the prefix if 'tracks' wasn't there
        sunspot_filename = f"{new_prefix}{base}"

    return ensure_extension(sunspot_filename, extension=".npz")


def sunspot_to_stats_filename(
        sunspot_path: str,
        quantities: list[Quantity]
) -> str:
    """
    Appends the processed quantities to the sunspot filename.
    Example: 'sunspots_Ic<0.9.npz' -> 'sunspots_Ic<0.9_Ic<B-Bp.npz'
    """
    base = path.basename(sunspot_path)
    stem, ext = path.splitext(base)  # Safely splits 'file' and '.npz'

    # Create the quantities string: 'Ic-B-Bp'
    q_str = f"{SEP_IN}".join(quantities)
    stats_filename = f"{stem}{SEP_OUT}{q_str}{ext}"

    return ensure_extension(stats_filename, extension=".npz")


def get_sunspot_phases_stem() -> str:
    """Returns fixed stem: 'sunspots_phases'"""
    return f"sunspots{SEP_OUT}phases"  # NO EXTENSION HERE
