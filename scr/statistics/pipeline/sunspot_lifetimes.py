from scr.utils.types_alias import Sunspot


def compute_lifetime(
        sunspot: Sunspot,
        region: str
) -> int:
    """
    Compute the lifetime of a single region as the number of frames
    with at least one non-empty contour.
    """
    region_frames = sunspot.get(region, {})
    lifetime = sum(
        bool(region_frames.get(frame, []))  # True if list is non-empty
        for frame in region_frames
    )
    return lifetime


def compute_lifetime_over_sunspot(
        sunspot: Sunspot,
) -> dict[str, int]:
    """Compute lifetime for all regions of a sunspot."""
    return {region: compute_lifetime(sunspot, region) for region in sunspot}
