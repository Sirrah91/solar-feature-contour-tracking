import numpy as np

from scr.utils.filesystem import is_empty
from scr.utils.types_alias import Sunspot

from scr.statistics.computation.masks import (
    compute_nesting_matrix,
    build_refined_masks,
    safe_subtract,
    overall_mask,
)

from scr.geometry.contours.densify import densify_contour
from scr.geometry.raster.api import rasterize


def prepare_level_data(
        *,
        sunspot: Sunspot,
        frame: int,
        shape: tuple[int, int],
        max_vertex_spacing: float,
) -> dict:
    """
    Precompute all reusable geometry per level with consistent float64 precision.
    """

    level_data = {}

    for level in sunspot:
        contours = sunspot[level].get(frame, [])
        if is_empty(contours):  # needed to create rings
            level_data[level] = {
                "contours": [],
                "signed_masks": [],
                "refined_masks": [],
                "total_mask": [],
                "total_area": 0.0,
                "densified_contours": [],
            }
            continue

        contours = [contour.astype(np.float64) for contour in contours]

        signed_masks = [
            rasterize(c, shape, mode="surface", engine="cairo", use_orientation=True, dtype=np.float64)
            for c in contours
            if not is_empty(c)
        ]

        nesting = compute_nesting_matrix(contours)

        refined_masks = build_refined_masks(
            contours=contours,
            individual_masks=signed_masks,
            nesting_matrix=nesting,
        )

        total = overall_mask(refined_masks)

        densified = [
            densify_contour(c, max_vertex_spacing=max_vertex_spacing)
            for c in contours
        ]

        level_data[level] = {
            "contours": contours,
            "signed_masks": signed_masks,
            "refined_masks": refined_masks,
            "total_mask": total,
            "total_area": np.nansum(total),
            "densified_contours": densified,
        }

    return level_data


def build_regions(level_data: dict) -> tuple[dict, str]:
    """
    Build all region types from prepared geometry.

    Parameters
    ----------
    level_data : dict
        Output of prepare_level_data()

    Returns
    -------
    dict
        Regions dictionary
    str
        Key of outermost region
    """

    regions = {}

    # 1. Filled Regions
    for level, data in level_data.items():
        if not is_empty(data["densified_contours"]):  # do not list empty regions
            regions[level] = {
                "masks": data["refined_masks"],
                "total_mask": data["total_mask"],
                "contours": data["densified_contours"],
                "total_contour": np.vstack(data["densified_contours"]) if not is_empty(data["densified_contours"]) else [],
                "levels": (level,),
                "is_ring": False,
            }

    # 2. Identify Outermost
    # Sort filled regions by area to find the truth
    _, outermost_key = _sort_and_finalize(regions)

    # 3. Ring Regions
    for level, base in level_data.items():
        for level2, inner in level_data.items():
            # THIS ALSO REMOVES RINGS OF REGIONS WHERE NONE BOUNDARY IS DEFINED -> NOT IN THE LIST OF REGIONS
            # if only inner level is missing, the regions equals to the filled one
            if (level == level2) or (np.nansum(safe_subtract(base["total_mask"], inner["total_mask"])) <= 0):
                continue

            # Flip Level 2 contours and masks to act as holes
            refined_ring = build_refined_masks(
                contours=base["contours"] + [contour[::-1] for contour in inner["contours"] if not is_empty(contour)],
                individual_masks=base["signed_masks"] + [-m for m in inner["signed_masks"]]
            )

            regions[f"{level}-{level2}"] = {
                "masks": refined_ring,
                "total_mask": overall_mask(refined_ring),
                "contours": [],
                "total_contour": [],
                "levels": (level, level2),
                "is_ring": True,
            }

    # 4. Special Regions (Holes and Envelope)
    if outermost_key:
        outer_data = level_data[outermost_key]

        # --- A. Internal Voids (The Holes) ---
        # We identify holes (area < 0) and turn them into positive regions
        void_contours, void_masks = [], []
        signed_areas = [np.nansum(m) for m in outer_data["signed_masks"]]

        # Build regions
        for c, m, a in zip(outer_data["densified_contours"], outer_data["signed_masks"], signed_areas):
            if not is_empty(c) and a < 0:  # Is hole
                void_contours.append(c[::-1])  # Flip orientation
                void_masks.append(-m)  # Flip polarity to positive

        if void_contours:
            # Build refined voids (handles islands inside the voids)
            refined_voids = build_refined_masks(contours=void_contours, individual_masks=void_masks)
            total_void_mask = overall_mask(refined_voids)
            total_void_contour = np.vstack(void_contours)
        else:
            refined_voids, total_void_mask, total_void_contour = [], [], []

        regions["internal_voids"] = {
            "masks": refined_voids,
            "total_mask": total_void_mask,
            "contours": void_contours,
            "total_contour": total_void_contour,
            "levels": (outermost_key, "void"),
            "is_ring": False,
        }

        # --- B. Sunspot Envelope (Solid Footprint) ---

        contours = outer_data["densified_contours"]
        signed_masks = outer_data["signed_masks"]

        # 1) detect nesting
        nesting = compute_nesting_matrix(contours)

        # 2) keep only outer contours (not nested inside any other)
        is_child = nesting.any(axis=1)
        outer_indices = np.where(~is_child)[0]

        refined_masks = []
        refined_contours = []

        for idx in outer_indices:

            contour = contours[idx]
            mask = signed_masks[idx]

            # 3) orient contour according to mask sign
            sign = np.sign(signed_areas[idx])

            if sign < 0:
                contour = contour[::-1]

            refined_masks.append(mask * sign)
            refined_contours.append(contour)

        # 4) merge masks
        total_mask = overall_mask(refined_masks)

        # 5) merge contours
        total_contour = np.vstack(refined_contours)

        regions["sunspot"] = {
            "masks": refined_masks,
            "total_mask": total_mask,
            "contours": refined_contours,
            "total_contour": total_contour,
            "levels": (outermost_key, "envelope"),
            "is_ring": False,
        }

    return _sort_and_finalize(regions)


def _sort_and_finalize(regions: dict) -> tuple[dict, str]:
    def key_func(k: str) -> tuple[int, float]:
        # Rank: Filled(0) > Rings(1) > Voids(2) > Sunspot(3)
        if regions[k]["is_ring"]:
            rank = 1
        elif k == "internal_voids":
            rank = 2
        elif k == "sunspot":
            rank = 3
        else:
            rank = 0  # filled regions

        area = regions[k].get("total_area")
        if area is None:
            mask = regions[k]["total_mask"]
            area = np.nansum(mask) if not is_empty(mask) else 0

        return rank, -area

    keys = sorted(regions.keys(), key=key_func)
    sorted_dict = {k: regions[k] for k in keys}
    outermost_key = next(  # first filled
        (k for k in keys if not regions[k]["is_ring"] and k not in ["internal_voids", "sunspot"]),
        None
    )

    return sorted_dict, outermost_key
