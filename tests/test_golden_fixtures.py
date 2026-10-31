"""Integrity tests for the Phase 1 golden fixtures (mock data only, no ``vl`` code).

Every analytic invariant is recomputed here from the raw positions with plain
finite differences, so the committed JSON is checked against the physics it claims.
"""

import filecmp
import json
import math
from pathlib import Path
from typing import Any

from fixtures import build_golden
from fixtures.loader import GoldenCase, list_golden_cases, load_golden, raw_dataframe

DT = 0.0625
CASES = [
    "g01_constant_speed",
    "g02_hard_brake",
    "g03_sharp_turn",
    "g04_stop",
    "g05_dropped_frame",
    "g06_position_jump",
    "g07_duplicate_frame",
    "g08_ts_backwards",
    "g09_sparse_cloud",
    "g10_gap_inside_brake",
    "g11_single_frame",
    "g12_empty",
]
RAW_KEYS = {"drive_id", "frame_idx", "ts", "n_points", "x", "y", "z"}
EXPECTED_KEYS = {
    "n_frames_out",
    "error_codes",
    "events",
    "no_event_contains_frames",
    "speed_checks",
    "yaw_rate_checks",
    "distance_m",
    "distance_tol",
    "null_kinematics_frames",
}
ERROR_CODES = {"VL-001", "VL-002", "VL-003", "VL-004", "VL-005", "VL-006"}
EVENT_TYPES = {"HARD_BRAKE", "HARD_ACCEL", "SHARP_TURN", "STOP"}


def _g(name: str) -> GoldenCase:
    """Load a case by its short prefix (for example ``"g01"``).

    Args:
        name: Case prefix such as ``"g01"``.

    Returns:
        The matching golden case.
    """
    full = next(c for c in CASES if c.startswith(name))
    return load_golden(full)


def _pos(case: GoldenCase) -> dict[int, tuple[float, float]]:
    """Map frame index to (x, y), keeping the first row of any duplicated frame.

    Args:
        case: The golden case.

    Returns:
        Positions keyed by frame index.
    """
    out: dict[int, tuple[float, float]] = {}
    for row in case.raw:
        out.setdefault(row["frame_idx"], (row["x"], row["y"]))
    return out


def _step(case: GoldenCase, a: int, b: int) -> tuple[float, float, float]:
    """Return (dx, dy, dt) between frames ``a`` and ``b`` using nominal timing.

    Args:
        case: The golden case.
        a: Earlier frame index.
        b: Later frame index.

    Returns:
        Displacement in x, y and the elapsed time ``(b - a) * DT``.
    """
    pos = _pos(case)
    return pos[b][0] - pos[a][0], pos[b][1] - pos[a][1], (b - a) * DT


def _fd_speed(case: GoldenCase, idx: int) -> float:
    """Finite-difference speed at a frame: forward step, else backward step.

    Args:
        case: The golden case.
        idx: Frame index.

    Returns:
        Speed in m/s from nominal frame timing.
    """
    frames = sorted(_pos(case))
    pos = frames.index(idx)
    a, b = (idx, frames[pos + 1]) if pos + 1 < len(frames) else (frames[pos - 1], idx)
    dx, dy, dt = _step(case, a, b)
    return math.hypot(dx, dy) / dt


def _heading(case: GoldenCase, a: int, b: int) -> float:
    """Heading of the chord from frame ``a`` to frame ``b``.

    Args:
        case: The golden case.
        a: Start frame index.
        b: End frame index.

    Returns:
        Heading in radians.
    """
    dx, dy, _ = _step(case, a, b)
    return math.atan2(dy, dx)


def _accel(case: GoldenCase, idx: int) -> float:
    """Centered second difference of the along-x position at ``idx``.

    Args:
        case: The golden case (straight-line motion along x only).
        idx: Frame index with neighbours at ``idx - 1`` and ``idx + 1``.

    Returns:
        Acceleration in m/s^2.
    """
    pos = _pos(case)
    return (pos[idx + 1][0] - 2 * pos[idx][0] + pos[idx - 1][0]) / DT**2


def _assert_events_valid(events: list[dict[str, Any]], name: str) -> None:
    """Check events are time-ordered, well-formed and use known types.

    Args:
        events: The expected ``events`` list.
        name: Case name for assertion messages.
    """
    starts = [e["start_frame"] for e in events]
    assert starts == sorted(starts), name
    assert all(e["event_type"] in EVENT_TYPES for e in events), name
    assert all(e["start_frame"] <= e["end_frame"] and e["frame_tol"] == 6 for e in events), name


def _check_timing_rules(name: str, case: GoldenCase) -> None:
    """Check ts, point count and y follow the common rules except where stated.

    Args:
        name: Case name for assertion messages.
        case: The golden case.
    """
    assert all(set(row) == RAW_KEYS for row in case.raw), name
    assert all(row["drive_id"] == "golden" and row["z"] == 0.75 for row in case.raw), name
    for row in case.raw:
        idx = row["frame_idx"]
        if not (name.startswith("g08") and idx == 25):
            assert row["ts"] == idx * DT, (name, idx)
        if not (name.startswith("g09") and idx == 25):
            assert row["n_points"] == 200, (name, idx)
        if not name.startswith("g03"):
            assert row["y"] == 0.0, (name, idx)


def _check_expected_references(name: str, case: GoldenCase) -> None:
    """Check expected frames exist in raw, codes and event types are valid, checks avoid nulls.

    Args:
        name: Case name for assertion messages.
        case: The golden case.
    """
    frames = {row["frame_idx"] for row in case.raw}
    exp: dict[str, Any] = case.expected
    assert "error_codes" in exp and "events" in exp, name
    assert exp["n_frames_out"] == len(frames), name
    assert {int(k) for k in exp["error_codes"]} <= frames, name
    assert set(exp["error_codes"].values()) <= ERROR_CODES, name
    nulls = set(exp.get("null_kinematics_frames", []))
    assert nulls <= {int(k) for k in exp["error_codes"]}, name
    checks = exp.get("speed_checks", []) + exp.get("yaw_rate_checks", [])
    assert {c["frame_idx"] for c in checks} <= frames - nulls, name
    _assert_events_valid(exp["events"], name)


def test_all_cases_load_with_valid_structure() -> None:
    """All twelve cases load, follow the common timing rules and reference real frames."""
    assert list_golden_cases() == CASES
    for name in CASES:
        case = load_golden(name)
        assert case.case == name and case.description, name
        assert set(case.expected) <= EXPECTED_KEYS, name
        _check_timing_rules(name, case)
        _check_expected_references(name, case)


def _check_expected_against_positions(name: str, case: GoldenCase) -> None:
    """Recompute every expected speed, yaw rate and distance from the raw positions.

    Args:
        name: Case name for assertion messages.
        case: The golden case.
    """
    for check in case.expected.get("speed_checks", []):
        speed = _fd_speed(case, check["frame_idx"])
        assert abs(speed - check["speed"]) <= check["tol"], (name, check, speed)
    for check in case.expected.get("yaw_rate_checks", []):
        i = check["frame_idx"]
        rate = (_heading(case, i, i + 1) - _heading(case, i - 1, i)) / DT
        assert abs(rate - check["yaw_rate"]) <= check["tol"], (name, check, rate)
    if "distance_m" in case.expected:
        frames = sorted(_pos(case))
        total = sum(
            math.hypot(*_step(case, a, b)[:2]) for a, b in zip(frames, frames[1:], strict=False)
        )
        assert abs(total - case.expected["distance_m"]) <= case.expected["distance_tol"], (
            name,
            total,
        )


def _check_g01(case: GoldenCase) -> None:
    """Constant speed: every finite-difference speed is 10 m/s along +x."""
    assert len(case.raw) == 60
    for i in range(59):
        assert abs(_fd_speed(case, i) - 10.0) < 1e-9, i
    assert all(row["x"] == 10 * row["ts"] for row in case.raw)


def _check_g02(case: GoldenCase) -> None:
    """Braking frames 64..96 decelerate at -5 m/s^2 between 15 and 5 m/s."""
    assert len(case.raw) == 160
    for i in range(65, 96):
        assert abs(_accel(case, i) - (-5.0)) < 1e-6, i
    assert abs(_fd_speed(case, 40) - 15.0) < 1e-9
    assert abs(_fd_speed(case, 150) - 5.0) < 1e-9
    event = case.expected["events"][0]
    assert (event["start_frame"], event["end_frame"]) == (round(4 / DT), round(6 / DT))


def _check_g03(case: GoldenCase) -> None:
    """Frames 32..80 sit on a radius-10 circle; straights keep heading 0 and 1.5 rad."""
    pos = _pos(case)
    assert len(case.raw) == 128
    for i in range(32, 81):
        assert abs(math.hypot(pos[i][0] - 10.0, pos[i][1] - 10.0) - 10.0) < 1e-9, i
    assert all(pos[i][1] == 0.0 for i in range(32))
    assert abs(_heading(case, 100, 101) - 1.5) < 1e-9
    assert abs(_heading(case, 80, 127) - 1.5) < 1e-9
    assert abs(_fd_speed(case, 120) - 5.0) < 1e-9
    event = case.expected["events"][0]
    assert (event["start_frame"], event["end_frame"]) == (round(2 / DT), round(5 / DT))


def _check_g04(case: GoldenCase) -> None:
    """Deceleration is -2.5 m/s^2 over frames 16..48; the car is stationary from 48."""
    pos = _pos(case)
    assert len(case.raw) == 96
    for i in range(17, 48):
        assert abs(_accel(case, i) - (-2.5)) < 1e-6, i
    assert all(pos[i][0] == pos[48][0] for i in range(48, 96))
    assert abs(pos[48][0] - 10.0) < 1e-9
    assert case.expected["events"][0]["end_frame"] == len(case.raw) - 1


SHAPE_CHECKS = {
    "g01_constant_speed": _check_g01,
    "g02_hard_brake": _check_g02,
    "g03_sharp_turn": _check_g03,
    "g04_stop": _check_g04,
}


def test_golden_physics_invariants() -> None:
    """Expected checks match finite differences of the raw positions; g01..g04 keep their shape."""
    for name in CASES:
        case = load_golden(name)
        _check_expected_against_positions(name, case)
        if name in SHAPE_CHECKS:
            SHAPE_CHECKS[name](case)


def _without_key(row: dict[str, Any], key: str) -> dict[str, Any]:
    """Copy a row, dropping one field.

    Args:
        row: A raw row.
        key: The field to drop.

    Returns:
        The row without ``key``.
    """
    return {k: v for k, v in row.items() if k != key}


def _rows_without(name: str, drop: int) -> list[dict[str, Any]]:
    """Rows of a case with one frame removed.

    Args:
        name: Case prefix.
        drop: Frame index to remove.

    Returns:
        The remaining raw rows.
    """
    return [r for r in _g(name).raw if r["frame_idx"] != drop]


def _same_but_key(raw: list[dict[str, Any]], key: str, at: int) -> None:
    """Assert ``raw`` is g01 except that one field of frame ``at`` differs.

    Args:
        raw: Rows of the derived case.
        key: The field allowed to differ.
        at: The frame index whose row may differ in ``key``.
    """
    base = _g("g01").raw
    assert [r for r in raw if r["frame_idx"] != at] == [r for r in base if r["frame_idx"] != at]
    assert _without_key(raw[at], key) == _without_key(base[at], key)


def _check_g05() -> None:
    assert _g("g05").raw == _rows_without("g01", 30) and len(_g("g05").raw) == 59
    checks = {c["frame_idx"]: c for c in _g("g05").expected["speed_checks"]}
    assert checks[31]["speed"] == 10.0 and checks[31]["tol"] == 0.2


def _check_g06() -> None:
    for row, ref in zip(_g("g06").raw, _g("g01").raw, strict=True):
        shift = 5.0 if row["frame_idx"] >= 30 else 0.0
        assert row["x"] == ref["x"] + shift, row["frame_idx"]
        assert _without_key(row, "x") == _without_key(ref, "x"), row["frame_idx"]
    text = _g("g06").description
    assert "frame 30" in text and "frame 29" in text and "50 m/s" in text


def _check_g07() -> None:
    raw = _g("g07").raw
    twice = [r for r in raw if r["frame_idx"] == 20]
    assert len(raw) == 61 and len(twice) == 2 and twice[0] == twice[1]
    assert [r for i, r in enumerate(raw) if i != 21] == _g("g01").raw


def _check_g08() -> None:
    raw = _g("g08").raw
    assert raw[25]["ts"] == raw[24]["ts"] - 0.001 < raw[24]["ts"]
    _same_but_key(raw, "ts", 25)


def _check_g09() -> None:
    raw = _g("g09").raw
    assert raw[25]["n_points"] == 10
    _same_but_key(raw, "n_points", 25)


def _check_g10() -> None:
    assert _g("g10").raw == _rows_without("g02", 80) and len(_g("g10").raw) == 159
    assert _g("g10").expected["no_event_contains_frames"] == [81]


def _check_g11_g12() -> None:
    only = _g("g11").raw
    assert len(only) == 1 and only[0]["frame_idx"] == 0 and only[0]["x"] == 0.0
    assert _g("g12").raw == []


def test_fault_cases_differ_from_their_base_exactly() -> None:
    """g05..g12 are g01/g02 with exactly one documented change each."""
    for check in (
        _check_g05, _check_g06, _check_g07, _check_g08, _check_g09, _check_g10, _check_g11_g12
    ):  # fmt: skip
        check()


def test_raw_dataframe_matches_the_json_rows() -> None:
    """The DataFrame has the raw columns and dtypes, an empty case keeps them, values match JSON."""
    df = raw_dataframe(_g("g01"))
    assert list(df.columns) == ["drive_id", "frame_idx", "ts", "n_points", "x", "y", "z"]
    assert len(df) == 60
    assert df["frame_idx"].dtype == "int64" and df["n_points"].dtype == "int64"
    assert all(df[c].dtype == "float64" for c in ("ts", "x", "y", "z"))
    empty = raw_dataframe(_g("g12"))
    assert empty.empty and list(empty.columns) == list(df.columns)
    assert empty.dtypes.equals(df.dtypes)
    name = "g05_dropped_frame"
    path = Path(build_golden.__file__).parent / "golden" / f"{name}.json"
    rows = json.loads(path.read_text())["raw"]
    five = raw_dataframe(load_golden(name))
    for column in RAW_KEYS:
        assert five[column].tolist() == [r[column] for r in rows], column


def test_builder_is_deterministic_and_matches_committed(tmp_path: Path) -> None:
    """Building twice is byte-identical and equals the committed fixture files."""
    one, two = tmp_path / "one", tmp_path / "two"
    build_golden.write_all(one)
    build_golden.write_all(two)
    committed = Path(build_golden.__file__).parent / "golden"
    names = sorted(p.name for p in committed.glob("*.json"))
    assert names == [f"{c}.json" for c in CASES]
    for name in names:
        assert filecmp.cmp(one / name, two / name, shallow=False), name
        assert filecmp.cmp(one / name, committed / name, shallow=False), name
