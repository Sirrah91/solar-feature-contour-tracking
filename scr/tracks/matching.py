import numpy as np
from skimage.morphology import dilation, disk
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import EuclideanTransform
from skimage.registration import phase_cross_correlation
import warnings
from typing import Literal

from scr.config.numerics import RND_SEED
from scr.utils.types_alias import Contour, Mask


def compute_iou(
        mask1: Mask,
        mask2: Mask,
        dilation_radius: int = 0
) -> float:
    """
    Compute the Intersection-over-Union (IoU) between two masks.
    Optional morphological dilation can help bridge small gaps.
    """
    if dilation_radius:
        selem = disk(dilation_radius)
        mask1 = dilation(mask1, selem)
        mask2 = dilation(mask2, selem)

    intersection = np.logical_and(mask1, mask2).sum()
    if intersection > 0.:
        union = np.logical_or(mask1, mask2).sum()
        if union > 0.:
            return intersection / union
    return 0.0


def warp_contour(
        contour: Contour,
        transform: EuclideanTransform
) -> Contour:
    """
    Apply Euclidean transformation to a contour.

    Parameters:
        contour: Input contour as (N, 2) array in (y, x) format.
        transform: Euclidean transformation to apply.

    Returns:
        Transformed contour of the same shape.
    """
    return transform(contour[:, ::-1])[:, ::-1]


def register_images_pairwise(
        img_target: np.ndarray,
        img_source: np.ndarray,
        residual_threshold_max: float = 3.5,
        qs_threshold: float = 0.7,
        qs_mask_direction: Literal["above", "below"] = "below",
        match_spatial_tolerance: float = 50.0,
        seed: int = RND_SEED,
) -> EuclideanTransform:
    """
    Estimate Euclidean transform that maps `img_source` onto `img_target`
    using ORB features and RANSAC, with fallback to phase correlation.

    Parameters:
        img_target: The reference image.
        img_source: The image to align.
        residual_threshold_max: Max residual for RANSAC.
        qs_threshold: Intensity threshold to mask granulation or magnetism.
        qs_mask_direction: "below" to keep quiet-Sun, "above" to keep active areas.
        match_spatial_tolerance: Max pixel distance allowed between matched keypoints.
        seed: Random number seed.

    Returns:
        A EuclideanTransform object mapping img_source to img_target.
    """

    def try_feature_registration() -> EuclideanTransform | None:
        orb = ORB(n_keypoints=2000, fast_threshold=0.08)

        try:
            orb.detect_and_extract(img_target)
            keypoints_target, descriptors_target = orb.keypoints, orb.descriptors
            orb.detect_and_extract(img_source)
            keypoints_source, descriptors_source = orb.keypoints, orb.descriptors
        except Exception as error:
            raise RuntimeError(f"ORB extraction failed: {error}")

        if len(keypoints_target) == 0 or len(keypoints_source) == 0:
            raise ValueError("No features found in one or both images.")

        matches = match_descriptors(descriptors_target, descriptors_source, cross_check=True)
        if len(matches) < 3:
            raise ValueError("Not enough matches for RANSAC.")

        src = keypoints_source[matches[:, 1]][:, ::-1]
        dst = keypoints_target[matches[:, 0]][:, ::-1]

        displacements = dst - src
        displacements -= np.mean(displacements, axis=0)  # always subtract mean (global) shift
        distances = np.linalg.norm(displacements, axis=1)
        close = distances < match_spatial_tolerance
        src, dst = src[close], dst[close]

        if len(src) < 3:
            raise ValueError("Not enough spatially close matches.")

        residual_threshold = 1.0
        while residual_threshold <= residual_threshold_max:
            model, _ = ransac((src, dst), EuclideanTransform,
                              min_samples=3,
                              residual_threshold=residual_threshold,
                              max_trials=1000,
                              rng=seed)
            if model and np.all(np.isfinite(model.params)):
                return model
            residual_threshold += 0.25

        raise ValueError("RANSAC failed to find a valid model.")

    def try_phase_correlation() -> EuclideanTransform:
        if qs_mask_direction == "below":
            mask_target = np.abs(img_target) < qs_threshold
            mask_source = np.abs(img_source) < qs_threshold
        else:
            mask_target = np.abs(img_target) > qs_threshold
            mask_source = np.abs(img_source) > qs_threshold

        shift, error, _ = phase_cross_correlation(
            reference_image=img_target,
            moving_image=img_source,
            reference_mask=mask_target,
            moving_mask=mask_source,
            upsample_factor=2
        )

        if not np.all(np.isfinite(shift)) or np.linalg.norm(shift) > max(img_target.shape) * 0.2:
            raise ValueError("Unreasonable shift detected.")

        return EuclideanTransform(translation=shift[::-1])  # yx → xy

    # --- Main logic ---
    try:
        return try_feature_registration()
    except Exception as e:
        warnings.warn(f"Feature registration failed ({e}); falling back to phase correlation.", RuntimeWarning)
        try:
            return try_phase_correlation()
        except Exception as e:
            warnings.warn(f"Phase correlation also failed ({e}); using identity transform.", RuntimeWarning)
            return EuclideanTransform()  # Identity
