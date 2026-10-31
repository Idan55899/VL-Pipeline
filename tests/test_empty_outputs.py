"""Review round 1, item 3: empty events and drives keep the DDL dtypes."""

import warnings

import pandas as pd

from pipeline_helpers import frames_table, run_pattern
from vl.etl.features import detect_events, summarize_drive

EVENTS_DTYPES = {
    "drive_id": "object",
    "event_type": "object",
    "start_frame": "int64",
    "end_frame": "int64",
    "start_ts": "float64",
    "end_ts": "float64",
    "params": "object",
}
DRIVES_DTYPES = {
    "drive_id": "object",
    "vehicle_id": "object",
    "vehicle_kind": "object",
    "drive_start_utc": "object",
    "start_ts": "float64",
    "duration_s": "float64",
    "n_frames": "int64",
    "distance_m": "float64",
    "n_events": "int64",
    "n_errors": "int64",
}


def dtypes_of(table: pd.DataFrame) -> dict[str, str]:
    """Return the dtype names of a table."""
    return {name: str(dtype) for name, dtype in table.dtypes.items()}


def one_event() -> pd.DataFrame:
    """Detect the single hard brake of a hand-made drive."""
    return detect_events(frames_table(40, accel=run_pattern(40, 10, 6, -4.0)))


def test_events_keep_the_events_dtypes_empty_or_not() -> None:
    assert dtypes_of(detect_events(frames_table(40))) == EVENTS_DTYPES
    assert dtypes_of(detect_events(frames_table(0))) == EVENTS_DTYPES
    assert dtypes_of(one_event()) == EVENTS_DTYPES


def test_drives_keep_the_drives_dtypes_empty_or_not() -> None:
    empty = summarize_drive(frames_table(0), detect_events(frames_table(0)))
    assert empty.empty
    assert dtypes_of(empty) == DRIVES_DTYPES
    frames = frames_table(60)
    assert dtypes_of(summarize_drive(frames, detect_events(frames))) == DRIVES_DTYPES


def test_concatenating_empty_and_non_empty_tables_keeps_dtypes_without_warning() -> None:
    frames = frames_table(60)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        events = pd.concat([detect_events(frames), one_event()], ignore_index=True)
        empty_drive = summarize_drive(frames_table(0), detect_events(frames))
        drives = pd.concat([empty_drive, summarize_drive(frames, events)], ignore_index=True)
    assert dtypes_of(events) == EVENTS_DTYPES
    assert dtypes_of(drives) == DRIVES_DTYPES
