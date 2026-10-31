"""Unit tests for detect_events on hand-made frame tables (thresholds, runs, params)."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pipeline_helpers import DT, events_of_type, frames_table, run_pattern, with_codes
from vl.config import DEFAULT, Config
from vl.etl.features import detect_events
from vl.schemas import EVENTS_COLUMNS, FRAMES_COLUMNS


def one_event(frames: pd.DataFrame, cfg: Config = DEFAULT) -> pd.DataFrame:
    """Detect events and assert there is exactly one."""
    events = detect_events(frames, cfg)
    assert len(events) == 1, f"events found:\n{events}"
    return events


def test_event_thresholds_are_inclusive_on_the_event_side() -> None:
    """Both sides of every boundary: hard brake <= -3.5, hard accel >= 3.0, |yaw| >= 0.35."""
    cases = [
        ("accel", -3.5, "HARD_BRAKE"),
        ("accel", -3.49, None),
        ("accel", 3.0, "HARD_ACCEL"),
        ("accel", 2.99, None),
        ("yaw_rate", 0.35, "SHARP_TURN"),
        ("yaw_rate", -0.35, "SHARP_TURN"),
        ("yaw_rate", 0.34, None),
    ]
    for column, value, detected in cases:
        events = detect_events(frames_table(40, **{column: run_pattern(40, 10, 6, value)}))
        assert events["event_type"].tolist() == ([detected] if detected else []), (column, value)


def test_event_params_carry_the_extreme_value_of_the_run() -> None:
    brake = np.zeros(40)
    brake[10:15] = [-4.0, -5.0, -4.2, -4.0, -4.1]
    yaw = np.zeros(40)
    yaw[20:24] = [0.4, 0.6, -0.7, 0.5]
    events = detect_events(frames_table(40, accel=brake, yaw_rate=yaw))
    brake_params = json.loads(events_of_type(events, "HARD_BRAKE").iloc[0]["params"])
    turn_params = json.loads(events_of_type(events, "SHARP_TURN").iloc[0]["params"])
    assert brake_params["min_accel"] == pytest.approx(-5.0)
    assert turn_params["max_abs_yaw_rate"] == pytest.approx(0.7)
    accel = np.zeros(40)
    accel[10:15] = [3.5, 4.0, 3.2, 3.0, 3.1]
    event = one_event(frames_table(40, accel=accel)).iloc[0]
    assert event["event_type"] == "HARD_ACCEL"
    assert json.loads(event["params"])["max_accel"] == pytest.approx(4.0)


def test_run_needs_min_event_frames_and_follows_the_config() -> None:
    for length, expected in [(3, 0), (4, 1)]:
        events = detect_events(frames_table(40, accel=run_pattern(40, 10, length, -4.0)))
        assert len(events) == expected, f"run of {length} frames"
    cfg = replace(DEFAULT, min_event_frames=2)
    frames = frames_table(40, accel=run_pattern(40, 10, 2, -4.0))
    assert len(detect_events(frames, cfg)) == 1


def test_event_frames_are_the_first_and_last_frame_of_the_run_inclusive() -> None:
    event = one_event(frames_table(40, accel=run_pattern(40, 10, 5, -4.0))).iloc[0]
    assert (event["start_frame"], event["end_frame"]) == (10, 14)
    assert event["start_ts"] == pytest.approx(10 * DT)
    assert event["end_ts"] == pytest.approx(14 * DT)
    assert event["drive_id"] == "d"


def test_runs_break_at_a_frame_with_an_error_code() -> None:
    frames = with_codes(frames_table(40, accel=run_pattern(40, 10, 10, -4.0)), [15], "VL-001")
    events = detect_events(frames).sort_values("start_frame")
    assert list(zip(events["start_frame"], events["end_frame"])) == [(10, 14), (16, 19)]


def test_stop_needs_more_than_31_frames_below_the_speed_threshold() -> None:
    for length, is_stop in [(20, False), (31, False), (32, True), (40, True)]:
        frames = frames_table(60, speed=run_pattern(60, 10, length, 0.0, base=10.0))
        events = detect_events(frames)
        assert events["event_type"].tolist() == (["STOP"] if is_stop else []), length
        if is_stop:
            duration = json.loads(events.iloc[0]["params"])["duration_s"]
            assert duration == pytest.approx(length * DT), length
    at_threshold = frames_table(60, speed=run_pattern(60, 10, 40, 0.2, base=10.0))
    assert detect_events(at_threshold).empty, "0.2 m/s is not strictly below the threshold"


def test_events_are_sorted_by_start_and_have_the_events_columns() -> None:
    accel = run_pattern(60, 5, 6, -4.0)
    yaw = run_pattern(60, 30, 6, 0.5)
    events = detect_events(frames_table(60, accel=accel, yaw_rate=yaw))
    assert list(events.columns) == EVENTS_COLUMNS
    assert events["event_type"].tolist() == ["HARD_BRAKE", "SHARP_TURN"]
    assert events["start_frame"].is_monotonic_increasing


def test_detect_events_on_empty_frames() -> None:
    events = detect_events(pd.DataFrame(columns=FRAMES_COLUMNS))
    assert events.empty
    assert list(events.columns) == EVENTS_COLUMNS


def test_null_kinematics_do_not_make_events() -> None:
    frames = frames_table(40, accel=np.full(40, np.nan), yaw_rate=np.full(40, np.nan))
    assert detect_events(frames).empty
