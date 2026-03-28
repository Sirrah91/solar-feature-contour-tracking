import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
from skimage.transform import EuclideanTransform

from scr.config.numerics import WP
from scr.utils.types_alias import Tracks, Events
from scr.utils.collections import nested_defaultdict
from scr.geometry.bbox import bboxes_intersect
from scr.geometry.contours.extraction import extract_frame_contours
from scr.geometry.contours.shapes import contour_to_shape, compute_iou
from scr.tracks.relabel import relabel_tracks_by_lifetime
from scr.tracks.matching import warp_contour, register_images_pairwise


def track_contours(
        images: np.ndarray,
        level: float,
        min_area: float = 5.,
        max_gap: int = 3,
        iou_threshold: float = 0.3,
        registration: bool = True,
        flip_contours: bool = False,
) -> tuple[Tracks, Events]:
    """
    Track contours across frames using IoU and image registration.

    Parameters:
        images: Array of shape (T, H, W), one image per time step.
        level: Contour level to extract.
        min_area: Minimum area (px) to consider a contour.
        max_gap: Max frames to look back for matches.
        iou_threshold: Minimum IoU to consider a match.
        registration: If True, register previous image to current.
        flip_contours: If True, it changes orientation of contours.

    Returns:
        Dictionary of tracks: {track_id: {frame_index: [contours]}}
    """

    tracks: Tracks = nested_defaultdict(factory=list, depth=2)
    next_id = 0
    registration_cache = {}
    events: Events = []

    for t, image in enumerate(tqdm(images, desc=f"Tracking contours at level {level}", unit="frame")):

        contours = extract_frame_contours(
            image=image,
            level=level,
            min_area=min_area
        )

        if flip_contours:
            contours = [c[::-1] for c in contours]

        curr_shapes = []
        for c in contours:
            shape = contour_to_shape(c)
            if isinstance(shape, Polygon) and shape.area > 0:
                curr_shapes.append(shape)
            else:
                curr_shapes.append(None)

        # -------------------------------------------------
        # Collect candidate previous states
        # -------------------------------------------------
        prev_entries = []   # (track_id, prev_area, prev_shape)

        for tid, hist in tracks.items():

            valid_frames = [
                tp for tp in hist.keys()
                if 0 < t - tp <= max_gap
            ]
            if not valid_frames:
                continue

            t_prev = max(valid_frames)

            pair_key = (t_prev, t)
            if pair_key not in registration_cache:
                registration_cache[pair_key] = (
                    register_images_pairwise(
                        img_source=images[t_prev].astype(WP),
                        img_target=image
                    ) if registration else EuclideanTransform()
                )

            transform = registration_cache[pair_key]

            for prev_c in hist[t_prev]:
                warped = warp_contour(prev_c, transform)
                shape = contour_to_shape(warped)
                if isinstance(shape, Polygon) and shape.area > 0:
                    prev_entries.append((tid, shape.area, shape))

        if not prev_entries:
            for c in contours:
                tracks[next_id][t].append(c)
                next_id += 1
            continue

        # -------------------------------------------------
        # Build overlap graph
        # -------------------------------------------------
        prev_to_curr = nested_defaultdict(factory=list)
        curr_to_prev = nested_defaultdict(factory=list)

        for i, (tid, prev_area, prev_shape) in enumerate(prev_entries):

            bbox_prev = prev_shape.bounds

            for j, curr_shape in enumerate(curr_shapes):
                if curr_shape is None:
                    continue

                bbox_curr = curr_shape.bounds

                if not bboxes_intersect(bbox_prev, bbox_curr):
                    continue

                if not prev_shape.intersects(curr_shape):
                    continue

                iou = compute_iou(prev_shape, curr_shape)

                if iou >= iou_threshold:
                    prev_to_curr[i].append(j)
                    curr_to_prev[j].append(i)

        assigned_curr = set()
        assigned_prev = set()

        # -------------------------------------------------
        # MERGES
        # -------------------------------------------------
        for j, prev_indices in curr_to_prev.items():

            if len(prev_indices) <= 1:
                continue

            areas = [prev_entries[i][1] for i in prev_indices]
            largest_idx = prev_indices[np.argmax(areas)]
            surviving_tid = prev_entries[largest_idx][0]

            tracks[surviving_tid][t].append(contours[j])

            parent_ids = [prev_entries[i][0] for i in prev_indices]

            events.append({
                "type": "merge",
                "frame": t,
                "parents": parent_ids,
                "survivor": surviving_tid,
            })

            assigned_curr.add(j)
            assigned_prev.update(prev_indices)

        # -------------------------------------------------
        # SPLITS
        # -------------------------------------------------
        for i, curr_indices in prev_to_curr.items():

            if i in assigned_prev:
                continue

            if len(curr_indices) <= 1:
                continue

            areas = [curr_shapes[j].area for j in curr_indices]
            largest_j = curr_indices[np.argmax(areas)]

            parent_tid = prev_entries[i][0]

            tracks[parent_tid][t].append(contours[largest_j])

            new_children = []

            for j in curr_indices:
                if j == largest_j:
                    continue
                tracks[next_id][t].append(contours[j])
                new_children.append(next_id)
                next_id += 1
                assigned_curr.add(j)

            events.append({
                "type": "split",
                "frame": t,
                "parent": parent_tid,
                "children": [parent_tid] + new_children,
            })

            assigned_prev.add(i)
            assigned_curr.add(largest_j)

        # -------------------------------------------------
        # 1-1 matches
        # -------------------------------------------------
        for i, curr_indices in prev_to_curr.items():

            if i in assigned_prev:
                continue

            if len(curr_indices) == 1:
                j = curr_indices[0]
                if j in assigned_curr:
                    continue

                tid = prev_entries[i][0]
                tracks[tid][t].append(contours[j])

                assigned_prev.add(i)
                assigned_curr.add(j)

        # -------------------------------------------------
        # New tracks
        # -------------------------------------------------
        for j, c in enumerate(contours):
            if j not in assigned_curr:
                tracks[next_id][t].append(c)
                next_id += 1

    tracks, events = relabel_tracks_by_lifetime(
        tracks=tracks,
        events=events
    )
    return tracks, events
