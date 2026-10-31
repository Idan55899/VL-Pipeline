"""Tests for derive_frames, detect_events and summarize_drive (golden and simulator driven)."""

import json
from typing import Any

import pandas as pd
import pytest

from fixtures.loader import GoldenCase, list_golden_cases, load_golden, raw_dataframe
from pipeline_helpers import DT, events_of_type, frames_table, raw_from_frames
from vl.etl.features import derive_frames, detect_events, summarize_drive
from vl.mock.drive_sim import SimConfig, simulate_drive
from vl.mock.faults import Fault
from vl.schemas import DRIVES_COLUMNS, EVENTS_COLUMNS, FRAMES_COLUMNS, FRAMES_DTYPES

CLEAN = SimConfig.clean()
METADATA = ["vehicle_id", "vehicle_kind", "drive_start_utc"]
KINEMATICS = ["vx", "vy", "speed", "accel", "yaw_rate"]
# Hand-derived length of the scripted drive: 60 + 180 + 20 + 25 + 15 + 40 + 150 + 56.25 m
# (accelerate, cruise, brake, cruise, turn, accelerate, cruise, decelerate to rest).
SCRIPTED_DISTANCE_M = 546.25


# --- helpers ------------------------------------------------------------------------------


def scripted_frames(faults: tuple[Fault, ...] = ()) -> pd.DataFrame:
    """Simulate the clean scripted drive with ``faults`` and derive its frames."""
    drive = simulate_drive("d", sim=CLEAN, faults=faults)
    return derive_frames(raw_from_frames(drive.frames))


def golden_frames(name: str) -> pd.DataFrame:
    """Derive the frames of a golden case."""
    return derive_frames(raw_dataframe(load_golden(name)))


def flagged_codes(frames: pd.DataFrame) -> dict[str, str]:
    """Map ``str(frame_idx)`` to the error code of every flagged frame."""
    flagged = frames[frames["error_code"].notna()]
    return {str(i): str(c) for i, c in zip(flagged["frame_idx"], flagged["error_code"])}


def frame_at(frames: pd.DataFrame, frame_idx: int) -> "pd.Series[Any]":
    """Return the single row of ``frames`` with the given ``frame_idx``."""
    rows = frames[frames["frame_idx"] == frame_idx]
    assert len(rows) == 1, f"expected exactly one row for frame {frame_idx}"
    return rows.iloc[0]


# --- golden-driven ------------------------------------------------------------------------


def check_events(name: str, events: pd.DataFrame, expected: list[dict[str, Any]]) -> None:
    """Assert the detected events match the expected ones within each ``frame_tol``."""
    found = events.sort_values("start_frame").reset_index(drop=True)
    assert len(found) == len(expected), f"{name}: events found:\n{found}"
    for row, want in zip(found.itertuples(), expected):
        assert row.event_type == want["event_type"], f"{name}: {want}"
        assert abs(row.start_frame - want["start_frame"]) <= want["frame_tol"], f"{name}: {want}"
        assert abs(row.end_frame - want["end_frame"]) <= want["frame_tol"], f"{name}: {want}"


def check_value_at_frames(
    name: str, frames: pd.DataFrame, column: str, checks: list[dict[str, Any]]
) -> None:
    """Assert ``column`` at each checked frame (``checks`` hold ``frame_idx``, value, ``tol``)."""
    for check in checks:
        value = frame_at(frames, check["frame_idx"])[column]
        assert value == pytest.approx(check[column], abs=check["tol"]), f"{name}: {check}"


def check_null_kinematics(frames: pd.DataFrame, frame_ids: list[int], name: str = "") -> None:
    """Assert suspect frames have no velocity, speed, acceleration and yaw rate."""
    for frame_idx in frame_ids:
        row = frame_at(frames, frame_idx)
        for column in KINEMATICS:
            assert pd.isna(row[column]), f"{name}: {column} of suspect frame {frame_idx}"


def check_no_event_contains(name: str, events: pd.DataFrame, frame_ids: list[int]) -> None:
    """Assert no event covers any of the given frames (both ends inclusive)."""
    for frame_idx in frame_ids:
        inside = (events["start_frame"] <= frame_idx) & (frame_idx <= events["end_frame"])
        assert not inside.any(), f"{name}: an event contains frame {frame_idx}:\n{events[inside]}"


def check_summary(
    name: str, frames: pd.DataFrame, events: pd.DataFrame, expected: dict[str, Any]
) -> None:
    """Assert the drive summary distance and counts for cases that pin ``distance_m``."""
    summary = summarize_drive(frames, events).iloc[0]
    assert summary["distance_m"] == pytest.approx(
        expected["distance_m"], abs=expected["distance_tol"]
    ), name
    assert summary["n_frames"] == expected["n_frames_out"], name
    assert summary["n_events"] == len(expected["events"]), name
    assert summary["n_errors"] == len(expected["error_codes"]), name


def check_golden_case(name: str) -> None:
    """Assert every expected key of one golden case, naming the case in each message."""
    case: GoldenCase = load_golden(name)
    expected = case.expected
    frames = derive_frames(raw_dataframe(case))
    events = detect_events(frames)
    assert (list(frames.columns), list(events.columns)) == (FRAMES_COLUMNS, EVENTS_COLUMNS), name
    assert len(frames) == expected["n_frames_out"], name
    assert flagged_codes(frames) == expected["error_codes"], name
    check_events(name, events, expected["events"])
    check_no_event_contains(name, events, expected.get("no_event_contains_frames", []))
    check_value_at_frames(name, frames, "speed", expected.get("speed_checks", []))
    check_value_at_frames(name, frames, "yaw_rate", expected.get("yaw_rate_checks", []))
    check_null_kinematics(frames, expected.get("null_kinematics_frames", []), name)
    if "distance_m" in expected:
        check_summary(name, frames, events, expected)


def test_golden_cases() -> None:
    """All 12 golden cases: codes, events, speeds, yaw, null kinematics, summary."""
    names = list_golden_cases()
    assert len(names) == 12, names
    for name in names:
        check_golden_case(name)


def test_event_params_carry_the_measured_extreme() -> None:
    cases = [
        ("g02_hard_brake", "HARD_BRAKE", "min_accel", -5.5, -3.5),  # true accel is -5
        ("g03_sharp_turn", "SHARP_TURN", "max_abs_yaw_rate", 0.35, 0.6),  # true rate is 0.5
        ("g04_stop", "STOP", "duration_s", 2.0, 3.1),  # stopped frames 48..95: 3.0 s
    ]
    for name, event_type, key, low, high in cases:
        events = detect_events(golden_frames(name))
        value = json.loads(events_of_type(events, event_type).iloc[0]["params"])[key]
        assert low <= value <= high, f"{name}: {key}={value} outside [{low}, {high}]"


def test_first_frame_has_null_speed_and_accel_and_yaw_is_zero_when_nearly_stopped() -> None:
    first = frame_at(golden_frames("g01_constant_speed"), 0)
    assert pd.isna(first["speed"])
    assert pd.isna(first["accel"])
    assert frame_at(golden_frames("g04_stop"), 90)["yaw_rate"] == 0.0


def test_unsorted_input_is_sorted_by_frame_idx() -> None:
    raw = raw_dataframe(load_golden("g01_constant_speed"))
    shuffled = raw.sample(frac=1.0, random_state=0)
    frames = derive_frames(shuffled)
    assert frames["frame_idx"].tolist() == list(range(60))
    assert flagged_codes(frames) == {}


def test_frame_after_backwards_ts_keeps_its_speed() -> None:
    frames = golden_frames("g08_ts_backwards")
    assert frame_at(frames, 26)["speed"] == pytest.approx(10.0, abs=0.3)
    assert "VL-003" not in set(frames["error_code"].dropna())


def sparse_shifted_raw() -> pd.DataFrame:
    """Build 60 frames at 10 m/s where frame 25 has 10 points and a centroid 0.5 m off."""
    raw = raw_dataframe(load_golden("g01_constant_speed"))
    row = raw["frame_idx"] == 25
    raw.loc[row, "n_points"] = 10
    raw.loc[row, "x"] += 0.5
    return raw


def test_sparse_frame_with_shifted_centroid_does_not_contaminate_neighbours() -> None:
    frames = derive_frames(sparse_shifted_raw())
    assert flagged_codes(frames) == {"25": "VL-006"}
    check_null_kinematics(frames, [25])
    for frame_idx in (23, 24, 26, 27, 28, 30):
        assert frame_at(frames, frame_idx)["speed"] == pytest.approx(10.0, abs=0.3)


def test_metadata_columns_pass_through_and_are_null_when_absent() -> None:
    drive = simulate_drive("d", sim=CLEAN, n_frames=40)
    frames = derive_frames(raw_from_frames(drive.frames))
    assert len(frames) > 0
    for column in METADATA:
        assert (frames[column] == drive.frames[0][column]).all(), column
    bare = golden_frames("g01_constant_speed")  # the golden raw rows carry no metadata
    assert len(bare) > 0
    for column in METADATA:
        assert column in bare.columns, column
        assert bare[column].isna().all(), column


# --- simulator-driven (SimConfig.clean) ----------------------------------------------------


def test_clean_drive_has_no_errors_the_scripted_events_and_speed() -> None:
    frames = scripted_frames()
    assert frames["error_code"].isna().all()
    events = detect_events(frames).sort_values("start_ts")
    assert events["event_type"].tolist() == ["HARD_BRAKE", "SHARP_TURN", "STOP"]
    assert abs(events.iloc[0]["start_ts"] - 20.0) <= 1.0
    assert json.loads(events.iloc[0]["params"])["min_accel"] == pytest.approx(-5.0, abs=0.2)
    assert events.iloc[-1]["end_frame"] == frames["frame_idx"].iloc[-1]
    assert frame_at(frames, 240)["speed"] == pytest.approx(15.0, abs=0.3)  # ts 15.0 = frame 240


def test_each_injected_fault_gives_its_code_on_the_expected_frame_only() -> None:
    cases = [
        ("drop", 100, {"101": "VL-001"}),  # the frame after the gap
        ("jump", 300, {"300": "VL-002"}),  # in particular no VL-003 anywhere
        ("duplicate", 200, {"200": "VL-004"}),
        ("sparse", 400, {"400": "VL-006"}),
    ]
    for kind, frame_idx, want in cases:
        frames = scripted_frames((Fault(kind, frame_idx),))
        assert flagged_codes(frames) == want, kind
        assert frames["frame_idx"].is_unique, f"{kind}: repeated frame_idx"


def test_event_not_spanning_gap() -> None:
    # The hard brake spans frames 320..352; the drop removes frame 336, flagging 337.
    # End frames are inclusive, so containment is start <= flagged <= end.
    frames = scripted_frames((Fault("drop", 336),))
    brakes = events_of_type(detect_events(frames), "HARD_BRAKE")
    assert flagged_codes(frames) == {"337": "VL-001"}
    assert len(brakes) >= 1
    check_no_event_contains("drop in brake", brakes, [337])


def test_summary_counts() -> None:
    faults = (Fault("drop", 100), Fault("duplicate", 700))
    frames = scripted_frames(faults)
    events = detect_events(frames)
    summary = summarize_drive(frames, events).iloc[0]
    assert summary["n_events"] == 3
    assert summary["n_errors"] == 2
    assert summary["n_frames"] == 999
    assert summary["distance_m"] == pytest.approx(SCRIPTED_DISTANCE_M, rel=0.02)


# --- edge cases ---------------------------------------------------------------------------


def test_single_frame_drive() -> None:
    frames = golden_frames("g11_single_frame")
    events = detect_events(frames)
    summary = summarize_drive(frames, events).iloc[0]
    assert len(frames) == 1
    assert pd.isna(frames.iloc[0]["speed"])
    assert events.empty
    assert summary["distance_m"] == 0
    assert summary["n_frames"] == 1


def test_empty_input() -> None:
    empty_inputs = {
        "no columns": pd.DataFrame(),
        "g12_empty": raw_dataframe(load_golden("g12_empty")),
    }
    for case, raw in empty_inputs.items():
        frames = derive_frames(raw)
        assert frames.empty, case
        assert list(frames.columns) == FRAMES_COLUMNS, case
        assert {name: str(dtype) for name, dtype in frames.dtypes.items()} == FRAMES_DTYPES, case
        assert detect_events(frames).empty, case


# --- summarize_drive ----------------------------------------------------------------------


def test_summary_is_one_row_with_drive_columns_and_metadata() -> None:
    frames = frames_table(60)
    summary = summarize_drive(frames, detect_events(frames))
    assert list(summary.columns) == DRIVES_COLUMNS
    assert len(summary) == 1
    row = summary.iloc[0]
    assert (row["drive_id"], row["vehicle_id"], row["vehicle_kind"]) == ("d", "veh-1", "car")
    assert row["drive_start_utc"] == "2026-01-01T00:00:00+00:00"
    assert row["start_ts"] == 0.0
    assert row["duration_s"] == pytest.approx(59 * DT)
    assert row["distance_m"] == pytest.approx(10.0 * 59 * DT, rel=0.02)


def test_summary_of_empty_frames_is_empty_with_drive_columns() -> None:
    summary = summarize_drive(
        pd.DataFrame(columns=FRAMES_COLUMNS), pd.DataFrame(columns=EVENTS_COLUMNS)
    )
    assert summary.empty
    assert list(summary.columns) == DRIVES_COLUMNS
