import numpy as np
from tqdm import tqdm

from scr.statistics.pipeline.single_frame import compute_frame_statistics

from scr.geometry.solar.mu import compute_mu
from scr.geometry.solar.projection import pixel_to_lonlat

from scr.io.fits.read import load_image, load_fits_headers

from scr.utils.types_alias import Sunspots, StatsByQuantity, Quantity
from scr.utils.collections import nested_defaultdict


def compute_sunspot_statistics_evolution(
        sunspots: Sunspots,
        image_paths: list[str],
        quantities: list[Quantity],
        max_vertex_spacing: float = 0.5,
        header_index: int = 0,
) -> StatsByQuantity:
    """
    Compute geometric and intensity-based statistics for regions of sunspots over time.
    Includes projection correction using mu.

    Statistics include:
    - Raw (pixel-based) area and perimeter
    - Flux-related quantities (sum and mean) within region and on boundary
    - Projection-corrected flux quantities using 1/mu weighting
    - Corrected geometric area and length based on 1/mu correction factor

    Parameters:
        sunspots: Dictionary of contours: {sid: {"region1": {t: [...]}, "region2": {t: [...]}}, ...}
        image_paths: List of images to be evaluated
        quantities: List of quantities to be evaluated.
        max_vertex_spacing: Maximum distance between contour points.
        header_index: Index in hdul where the header is read.

    Returns:
        Nested dictionary: {sid: {"region1": {t: {...}}, "region2": {...}, ..., "sunspot": {...}}}
    """

    # Result: stats[quantity][sid][region][t]
    final_stats = nested_defaultdict(factory=dict, depth=4)

    # 1. Map which sunspots are active in which frames
    # frame_to_sids = {t: [sid1, sid2, ...]}
    frame_to_sids = nested_defaultdict(factory=list, depth=1)
    for sid, sunspot in sunspots.items():
        # Get all frames across all levels for this specific sunspot
        active_frames = set().union(*[sunspot[lvl].keys() for lvl in sunspot])
        for frame in active_frames:
            frame_to_sids[frame].append(sid)

    # 2. Iterate through time (The heavy lifting)
    all_active_frames = sorted(frame_to_sids.keys())

    for frame in tqdm(all_active_frames, desc="Processing Frames", unit="frame"):
        # --- A. Load context once per frame ---
        # Wrap image_paths[frame] in a list so load_fits_headers behaves correctly
        headers_list = load_fits_headers([image_paths[frame]], header_index=header_index)
        header = headers_list[0]

        image_map = {q: load_image(image_paths[frame], q) for q in quantities}

        for q in ["Bp", "Bt"]:
            if q in image_map:
                image_map[q] = np.abs(image_map[q])

        # Precompute geometry once for the whole frame
        inv_mu2D = (1.0 / compute_mu(header)).astype(np.float64)
        lon2D, lat2D = pixel_to_lonlat(header)
        lon2D, lat2D = lon2D.astype(np.float64), lat2D.astype(np.float64)

        geo_context = {
            "inv_mu2D": inv_mu2D,
            "lon2D": lon2D,
            "lat2D": lat2D,
            "rsun": header["RSUN_OBS"] / header["CDELT1"],
            "shape": image_map[next(iter(image_map))].shape
        }

        # --- B. Process all sunspots active at this time ---
        for sid in frame_to_sids[frame]:
            sunspot = sunspots[sid]

            # Compute stats (This still needs to run per-sunspot because contours differ)
            frame_stats = compute_frame_statistics(
                sunspot=sunspot,
                frame=frame,
                image_map=image_map,
                geo_context=geo_context,
                max_vertex_spacing=max_vertex_spacing,
            )

            # 4. Store results
            for q in quantities:
                for r_key, region_data in frame_stats[q].items():
                    final_stats[q][sid][r_key][frame] = region_data

    return final_stats
