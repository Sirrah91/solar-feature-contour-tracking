from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from scr.contours.selection import select_support_contours
from scr.geometry.contours.extraction import find_contours
from scr.geometry.raster.api import rasterize
from scr.io.fits.read import load_image
from scr.physics.magnetic import compute_Binc, compute_unsigned_inclination
from scr.utils.filesystem import is_empty
from scr.utils.collections import nested_defaultdict
from scr.utils.types_alias import Contours, Mask, SunspotPhase, SunspotsPhasesByObservation


def _make_mask(contours: Contours, image_shape: tuple[int, int]) -> Mask:
    return rasterize(
        contours,
        image_shape,
        mode="surface",
        engine="skimage",
        use_orientation=False,
        dtype=bool,
    )


def extract_pixel_data(
        sunspots_phases: SunspotsPhasesByObservation,
        df: pd.DataFrame,
        levels: Sequence[float],
        *,
        phase: SunspotPhase,
) -> dict:
    """
    Extract per-pixel magnetic and intensity data for all spots in df.

    Iterates over all image paths in df, loads the required images,
    builds region masks for each sunspot, and accumulates pixel-level
    arrays. The returned dict can be saved directly with save_npz.

    Parameters
    ----------
    sunspots_phases : nested sunspot phase structure from load_filtered_phase_tracks
    df : filtered DataFrame with columns image_path, observation_id,
         frame, sunspot_id, spot_global_index, Br_sunspot_flux_corr_total,
         sunspot_area_corr
    levels : magnetic field contour levels [G] defining B-threshold regions
    phase : evolution phase ("forming", "stable", "decaying")

    Returns
    -------
    Nested dict: data[region][quantity] = list of per-object pixel arrays
    """
    data = nested_defaultdict(depth=2, factory=list)

    for image_path, group in tqdm(
            df.groupby(["image_path"], observed=True), desc="Images"
    ):
        image_path = image_path[0]

        images = {
            "Ic": load_image(image_path, quantity="Ic"),
            "B": load_image(image_path, quantity="B"),
            "Bhor": load_image(image_path, quantity="Bhor"),
            "Br": load_image(image_path, quantity="Br"),
        }
        images["Binc"] = compute_unsigned_inclination(
            signed_inclination=compute_Binc(
                Br=images["Br"], Bhor=images["Bhor"]
            )
        ).astype(np.float32)

        image_shape = images["B"].shape
        obs_id = group["observation_id"].iloc[0]
        frame = group["frame"].iloc[0]
        sunspots = sunspots_phases[obs_id]

        for spot_id, sunspot in sunspots.items():
            if phase not in sunspot:
                continue

            row = group[group["sunspot_id"] == spot_id]
            if is_empty(row):
                continue

            phi = row["Br_sunspot_flux_corr_total"].iloc[0]
            area = row["sunspot_area_corr"].iloc[0]
            obj_id = row["spot_global_index"].iloc[0]

            spot = sunspot[phase]

            penumbra_contours = spot.get("Ic<0.9", {}).get(frame, [])
            pore_contours = spot.get("Ic<0.65", {}).get(frame, [])
            umbra_contours = spot.get("Ic<0.5", {}).get(frame, [])

            region_masks = {
                "ic0.9": _make_mask(penumbra_contours, image_shape),
                "ic0.65": _make_mask(pore_contours, image_shape),
                "ic0.5": _make_mask(umbra_contours, image_shape),
            }

            for level in levels:
                b_contours = find_contours(images["B"], level=level)
                b_contours = select_support_contours(penumbra_contours, b_contours)
                region_masks[f"b{level}"] = _make_mask(b_contours, image_shape)

            for region, mask in region_masks.items():
                n = mask.sum()
                for q, img in images.items():
                    data[region][q].append(img[mask])
                data[region]["Phi"].append(np.full(n, phi))
                data[region]["area"].append(np.full(n, area))
                data[region]["obj_id"].append(np.full(n, obj_id))

    return data
