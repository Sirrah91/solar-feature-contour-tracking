import re
from astropy.time import Time
from typing import Literal

from scr.utils.time import parse_datetime


_SCALE_RE = re.compile(r"_(TAI|UTC|TT)$")


def parse_time_to_astropy(
        time_str: str,
        default_scale: Literal["utc", "tai", "tt"] = "utc",
) -> Time:
    """
    Parse FITS / JSOC time strings into an Astropy Time object.
    """

    t = time_str.strip()
    scale = default_scale

    m = _SCALE_RE.search(t)
    if m:
        scale = m.group(1).lower()
        t = t[: m.start()]

    dt = parse_datetime(t)
    if dt is None:
        raise ValueError(f"Unparseable time string: {time_str}")

    return Time(dt, scale=scale)
