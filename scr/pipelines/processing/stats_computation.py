from scr.utils.types_alias import StatsByQuantity, Quantity, Metadata, Events

from scr.io.sunspots import load_sunspot_file

from scr.statistics.pipeline.evolution import compute_sunspot_statistics_evolution


def compute_stats_from_sunspots(
        sunspot_path: str,
        quantities: list[Quantity],
        header_index: int = 0,
        max_vertex_spacing: float = 0.5,
) -> tuple[dict, StatsByQuantity, Metadata, Events]:
    """
    Returns: tracks, statistics, metadata, events
    """
    sunspots, _, metadata, events = load_sunspot_file(sunspot_path)

    image_paths = metadata["image_paths"]

    stats = compute_sunspot_statistics_evolution(
        sunspots=sunspots,
        image_paths=image_paths,
        quantities=quantities,
        max_vertex_spacing=max_vertex_spacing,
        header_index=header_index,
    )

    metadata |= {
        "sunspot_path": sunspot_path,
        "quantities": quantities,
        "header_index": header_index,
        "max_vertex_spacing": max_vertex_spacing,
    }

    return sunspots, stats, metadata, events
