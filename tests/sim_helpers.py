"""Shared helpers for the simulator tests (frames, centroids, finite differences)."""

from typing import Any

import numpy as np
import numpy.typing as npt

from vl.config import DEFAULT

DT = DEFAULT.frame_period_s
Frame = dict[str, Any]


def centroid_cm(frame: Frame) -> npt.NDArray[np.float64]:
    """Return the ``x, y, z`` centroid of a frame's points in centimeters."""
    return np.asarray(frame["points"], dtype=np.float64).mean(axis=0)


def centroids_cm(frames: list[Frame]) -> npt.NDArray[np.float64]:
    """Return an ``(n, 3)`` array of per-frame centroids in centimeters."""
    return np.array([centroid_cm(frame) for frame in frames])


def make_frames(n: int = 20, drive_id: str = "d") -> list[Frame]:
    """Build ``n`` hand-made clean frames with 60 distinct points each."""
    return [
        {
            "drive_id": drive_id,
            "vehicle_id": "veh-1",
            "vehicle_kind": "car",
            "drive_start_utc": "2026-01-01T00:00:00+00:00",
            "frame_idx": i,
            "ts": i * DT,
            "points": [[i * 100 + k, k, 0] for k in range(60)],
        }
        for i in range(n)
    ]


def frame_indices(frames: list[Frame]) -> list[int]:
    """Return the ``frame_idx`` of every frame, in emission order."""
    return [frame["frame_idx"] for frame in frames]
