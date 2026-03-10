import numpy as np

from scr.statistics.pipeline.sunspot_lifetimes import compute_lifetime

from scr.statistics.computation.regions import prepare_level_data, build_regions
from scr.statistics.computation.mask_stats import compute_mask_geometric_stats, compute_mask_intensity_stats
from scr.statistics.computation.contour_stats import compute_contour_geometric_stats, compute_contour_intensity_stats
from scr.statistics.aggregations import flatten_region_stats

from scr.utils.types_alias import Sunspot, Stat, Quantity, SunspotPart
from scr.utils.filesystem import is_empty
from scr.utils.collections import nested_defaultdict


def compute_frame_statistics(
        *,
        sunspot: Sunspot,
        frame: int,
        image_map: dict[Quantity, np.ndarray],
        geo_context: dict,
        max_vertex_spacing: float,
) -> dict[Quantity, dict[SunspotPart, Stat]]:
    """
    Computes geometry once, then maps intensity stats for all quantities.
    Returns: {quantity: {region_key: {flattened_stats}}}
    """
    # Extract context from cache
    inv_mu2D = geo_context["inv_mu2D"]
    lon2D = geo_context["lon2D"]
    lat2D = geo_context["lat2D"]
    rsun = geo_context["rsun"]
    shape = geo_context["shape"]

    # --- 2. Build Geometry (Once per Sunspot-Frame) ---
    level_data = prepare_level_data(
        sunspot=sunspot,
        frame=frame,
        shape=shape,
        max_vertex_spacing=max_vertex_spacing,
    )

    regions, outermost_key = build_regions(level_data)

    # Initialize results container
    # results[quantity][region_key] = {stats}
    results = nested_defaultdict(factory=dict, depth=2)

    # --- Compute Stats ---
    void_empty = (
            is_empty(regions["internal_voids"]["total_mask"])
            or np.nansum(regions["internal_voids"]["total_mask"]) == 0
    )

    for region_key, region in regions.items():

        do_full_compute = True

        # ---------------------------------------
        # RING OPTIMIZATION
        # ---------------------------------------

        if region["is_ring"]:
            outer, inner = region["levels"]

            inner_mask = regions[inner]["total_mask"]
            inner_empty = (
                    is_empty(inner_mask)
                    or np.nansum(inner_mask) == 0
            )

            if inner_empty:
                # Check if 'outer' exists for all quantities before committing to a copy
                can_copy = all(outer in results[q] for q in image_map)

                if can_copy:
                    for q in image_map:
                        results[q][region_key] = results[q][outer].copy()

                    do_full_compute = False

        # ---------------------------------------
        # SUNSPOT OPTIMIZATION
        # ---------------------------------------

        elif region_key == "sunspot":
            if void_empty:
                can_copy = all(outermost_key in results[q] for q in image_map)

                if can_copy:
                    for q in image_map:
                        results[q][region_key] = results[q][outermost_key].copy()

                    do_full_compute = False

        if not do_full_compute:
            continue

        # --- A. GEOMETRIC STATS (Computed Once) ---
        geo_mask = compute_mask_geometric_stats(
            masks=region["masks"],
            total_mask=region["total_mask"],
            projection_weights=inv_mu2D,
        )
        geo_cont = compute_contour_geometric_stats(
            contours=region["contours"],
            total_contour=region["total_contour"],
            lon2D=lon2D,
            lat2D=lat2D,
            rsun=rsun
        )

        # We need to keep arc_lengths for the flux computation, but not in final dict
        shared_arc_lengths = geo_cont.get("_arc_lengths")

        region_lifetime = compute_lifetime(sunspot, region_key)

        # --- B. INTENSITY STATS (Per Quantity) ---
        for q, image in image_map.items():
            image = image.astype(np.float64)

            int_mask = compute_mask_intensity_stats(
                masks=region["masks"],
                total_mask=region["total_mask"],
                image=image,
                projection_weights=inv_mu2D,
            )
            int_cont = compute_contour_intensity_stats(
                contours=region["contours"],
                image=image,
                inv_mu2D=inv_mu2D,
                arc_lengths=shared_arc_lengths,
            )

            # Consolidate and Flatten
            # We filter out the private '_arc_lengths' before flattening
            geo_cont_clean = {k: v for k, v in geo_cont.items() if not k.startswith("_")}

            flat = flatten_region_stats(
                {**geo_mask, **int_mask},
                {**geo_cont_clean, **int_cont}
            )

            flat["lifetime"] = np.int32(region_lifetime)
            results[q][region_key] = flat

    return results
