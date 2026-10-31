"""Shared helpers for the pipeline tests: raw rows from messages and hand-made frame tables."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from fixtures.loader import RAW_DTYPES as LOADER_RAW_DTYPES
from vl.config import DEFAULT
from vl.schemas import FRAMES_COLUMNS, FRAMES_DTYPES, RAW_FRAMES_COLUMNS

DT = DEFAULT.frame_period_s
ColumnValue = float | str | None | npt.NDArray[np.float64]
# The golden loader's raw dtypes plus the optional string metadata columns.
RAW_DTYPES = {c: LOADER_RAW_DTYPES.get(c, "object") for c in RAW_FRAMES_COLUMNS}


def raw_row(frame: dict[str, Any]) -> dict[str, Any]:
    """Convert one frame message into a raw row (centroid in meters, point count)."""
    points = np.asarray(frame["points"], dtype=np.float64)
    x, y, z = points.mean(axis=0) / 100.0
    row = {key: frame.get(key) for key in ("drive_id", "vehicle_id", "vehicle_kind")}
    row["drive_start_utc"] = frame.get("drive_start_utc")
    row.update(frame_idx=frame["frame_idx"], ts=frame["ts"], n_points=len(points), x=x, y=y, z=z)
    return row


def raw_from_frames(frames: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the raw DataFrame of one drive from its frame messages.

    Messages whose ``points`` is not a list (malformed) are skipped, as the ingestion
    would send them to ``rejects``.

    Args:
        frames: Frame messages in emission order.

    Returns:
        A DataFrame with ``RAW_FRAMES_COLUMNS``, one row per valid message.
    """
    rows = [raw_row(frame) for frame in frames if isinstance(frame["points"], list)]
    return pd.DataFrame(rows, columns=RAW_FRAMES_COLUMNS).astype(RAW_DTYPES)


def frames_table(n: int, **overrides: ColumnValue) -> pd.DataFrame:
    """Build a hand-made ``frames`` table: 10 m/s straight, no accel, no yaw, no error codes.

    Args:
        n: Number of rows (``frame_idx`` 0..n-1, ``ts = frame_idx * DT``).
        **overrides: Column values (scalar or array) replacing the defaults.

    Returns:
        A DataFrame with ``FRAMES_COLUMNS`` and the dtypes of ``FRAMES_DTYPES``.
    """
    ts = np.arange(n) * DT
    table = pd.DataFrame(
        {
            "drive_id": "d",
            "vehicle_id": "veh-1",
            "vehicle_kind": "car",
            "drive_start_utc": "2026-01-01T00:00:00+00:00",
            "frame_idx": np.arange(n),
            "ts": ts,
            "x": 10.0 * ts,
            "y": 0.0,
            "z": 0.75,
            "vx": 10.0,
            "vy": 0.0,
            "speed": 10.0,
            "accel": 0.0,
            "yaw_rate": 0.0,
            "error_code": None,
        },
        index=range(n),
    )
    for key, value in overrides.items():
        table[key] = value
    return table[FRAMES_COLUMNS].astype(FRAMES_DTYPES)


def run_pattern(
    n: int, start: int, length: int, value: float, base: float = 0.0
) -> npt.NDArray[np.float64]:
    """Build an array of ``n`` values equal to ``base`` with ``length`` values from ``start``."""
    values = np.full(n, base, dtype=np.float64)
    values[start : start + length] = value
    return values


def events_of_type(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """Return the events of one type, in time order."""
    return events[events["event_type"] == event_type].sort_values("start_frame")


def with_codes(table: pd.DataFrame, rows: list[int], code: str) -> pd.DataFrame:
    """Return a copy of ``table`` with ``error_code`` set to ``code`` at the given row positions."""
    out = table.copy()
    out["error_code"] = out["error_code"].astype(object)
    out.iloc[rows, out.columns.get_loc("error_code")] = code
    return out
