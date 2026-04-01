import numpy as np
import pandas as pd
from typing import Iterable, Sequence


def _linspace_frames(
        frames: Sequence[int],
        n: int
) -> list[int]:
    """
    Select n approximately evenly spaced frames from a sorted frame list,
    **excluding the first and last frames** to avoid edges.

    If n >= len(frames), returns all frames.
    """
    if n <= 0:
        return []
    if len(frames) <= n:
        return list(frames)

    # Add two "virtual" endpoints, then drop them to avoid edges
    positions = np.linspace(0, len(frames) - 1, n + 2)[1:-1]
    indices = np.round(positions).astype(int)
    return [frames[i] for i in indices]


def select_representative_frames(
        df: pd.DataFrame,
        phases: Iterable[str] = ("forming", "stable", "decaying"),
        n_frames: int = 3,
) -> list[int]:
    """
    Select exactly `n_frames` representative frame indices from a sunspot
    time series, prioritising phase medians and falling back to evenly spaced
    frames from the longest available phase.

    Rules
    -----
    1. If all phases are present: take one median frame per phase.
    2. If fewer phases are present:
       - take medians of available phases
       - recompute frames from the longest phase using linear spacing
         so that total count == n_frames
    3. Returned frames are always sorted by frame index.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ["phase", "frame"].
    phases : iterable of str
        Phase names to consider.
    n_frames : int
        Number of frames to return (default: 3).

    Returns
    -------
    list[int]
        Sorted list of selected frame indices.
    """
    frames_per_phase: dict[str, list[int]] = {}

    # Collect sorted frames per available phase
    for phase in phases:
        phase_frames = np.sort(df.loc[df["phase"] == phase, "frame"].to_numpy())
        if len(phase_frames) > 0:
            frames_per_phase[phase] = phase_frames.tolist()

    if not frames_per_phase:
        return []

    # Case 1: all phases available
    if len(frames_per_phase) >= n_frames:
        selected = [
            frames[len(frames) // 2]
            for frames in frames_per_phase.values()
        ]
        return sorted(selected[:n_frames])

    # Case 2 & 3: fewer phases than needed
    # First take medians
    selected_by_phase: dict[str, list[int]] = {}
    for phase, frames in frames_per_phase.items():
        selected_by_phase[phase] = [frames[len(frames) // 2]]

    # How many frames still needed?
    missing = n_frames - len(selected_by_phase)

    if missing > 0:
        # Find the longest phase
        longest_phase = max(
            frames_per_phase,
            key=lambda p: len(frames_per_phase[p]),
        )
        long_frames = frames_per_phase[longest_phase]

        # Recompute frames for this phase so that total == n_frames
        # (replace previous selection for this phase)
        n_from_longest = 1 + missing
        selected_by_phase[longest_phase] = _linspace_frames(
            long_frames,
            n_from_longest,
        )

    # Flatten + sort chronologically
    selected_frames = [
        frame
        for frames in selected_by_phase.values()
        for frame in frames
    ]

    return sorted(selected_frames)
