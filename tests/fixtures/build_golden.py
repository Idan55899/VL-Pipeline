"""Generate the golden mock drives and their analytic expectations.

Role in the pipeline: test support only. Run with
``.venv/bin/python -m tests.fixtures.build_golden`` to rewrite ``golden/*.json``.

Every position is a closed-form function of time (derivation next to each formula);
expectations are derived by hand from the same physics. Nothing here imports or runs
any code under ``src/vl``.

Import note: pytest puts ``tests/`` on ``sys.path`` so the tests import this module as
``fixtures.build_golden``, while the generator is run as ``tests.fixtures.build_golden``;
neither this module nor the loader imports the other, so both spellings work.
"""

import copy
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

DT = 0.0625  # 16 fps frame period in seconds (exact in binary floating point)
N_POINTS = 200
Z = 0.75
DRIVE_ID = "golden"
GOLDEN_DIR = Path(__file__).parent / "golden"
FRAME_TOL = 6

Row = dict[str, Any]
Position = Callable[[float], tuple[float, float]]


def frame_row(idx: int, x: float, y: float = 0.0, n_points: int = N_POINTS) -> Row:
    """Build one raw row; ``ts`` is exactly ``idx * DT``.

    Args:
        idx: Frame index.
        x: Centroid x in meters.
        y: Centroid y in meters.
        n_points: Number of cloud points.

    Returns:
        The raw row dictionary.
    """
    return {
        "drive_id": DRIVE_ID,
        "frame_idx": idx,
        "ts": idx * DT,
        "n_points": n_points,
        "x": x,
        "y": y,
        "z": Z,
    }


def sample(count: int, position: Position) -> list[Row]:
    """Sample a position function at frames ``0..count-1``.

    Args:
        count: Number of frames.
        position: Maps time in seconds to ``(x, y)`` in meters.

    Returns:
        The raw rows.
    """
    return [frame_row(i, *position(i * DT)) for i in range(count)]


def constant_speed(t: float) -> tuple[float, float]:
    """Position for 10 m/s along +x.

    Derivation: v = 10 constant, so x = 10 t. Speed per frame is 10 * DT / DT = 10.

    Args:
        t: Time in seconds.

    Returns:
        ``(x, y)`` in meters.
    """
    return 10.0 * t, 0.0


def hard_brake(t: float) -> tuple[float, float]:
    """Position for cruise, a -5 m/s^2 brake, then a slower cruise.

    Derivation: t < 4 is 15 m/s, so x = 15 t and x(4) = 60. For 4 <= t < 6 with
    tau = t - 4: v = 15 - 5 tau, x = 60 + 15 tau - 2.5 tau^2, so v(6) = 15 - 10 = 5
    and x(6) = 60 + 30 - 10 = 80. For t >= 6: x = 80 + 5 (t - 6). The brake spans
    frames 64 (t = 4) to 96 (t = 6).

    Args:
        t: Time in seconds.

    Returns:
        ``(x, y)`` in meters.
    """
    if t < 4.0:
        return 15.0 * t, 0.0
    if t < 6.0:
        tau = t - 4.0
        return 60.0 + 15.0 * tau - 2.5 * tau * tau, 0.0
    return 80.0 + 5.0 * (t - 6.0), 0.0


TURN_V = 5.0
TURN_OMEGA = 0.5
TURN_R = TURN_V / TURN_OMEGA  # r = v / omega = 5 / 0.5 = 10 m
TURN_X0 = TURN_V * 2.0  # straight at 5 m/s for t < 2, so x0 = 10 m
TURN_END = 5.0  # t = 5 is frame 80
TURN_THETA = TURN_OMEGA * (TURN_END - 2.0)  # 0.5 * 3 = 1.5 rad final heading


def sharp_turn(t: float) -> tuple[float, float]:
    """Position for a straight, a constant-yaw-rate arc, then a straight.

    Derivation: t < 2 is straight, x = 5 t, x0 = 10 at t = 2 (frame 32). For
    2 <= t < 5, theta = omega (t - 2) and the car moves on a circle of radius
    r = v / omega = 10 centered at (x0, r): x = x0 + r sin(theta),
    y = r (1 - cos(theta)). At t = 5 (frame 80) theta = 1.5 rad. For t >= 5 the car
    goes straight along heading 1.5: p = p(5) + v (t - 5) (cos 1.5, sin 1.5).

    Args:
        t: Time in seconds.

    Returns:
        ``(x, y)`` in meters.
    """
    if t < 2.0:
        return TURN_V * t, 0.0
    if t < TURN_END:
        theta = TURN_OMEGA * (t - 2.0)
        return TURN_X0 + TURN_R * math.sin(theta), TURN_R * (1.0 - math.cos(theta))
    x5 = TURN_X0 + TURN_R * math.sin(TURN_THETA)
    y5 = TURN_R * (1.0 - math.cos(TURN_THETA))
    run = TURN_V * (t - TURN_END)
    return x5 + run * math.cos(TURN_THETA), y5 + run * math.sin(TURN_THETA)


def stop(t: float) -> tuple[float, float]:
    """Position for cruise, a -2.5 m/s^2 deceleration to rest, then stopped.

    Derivation: t < 1 is 5 m/s, x = 5 t, x(1) = 5. For 1 <= t < 3 with tau = t - 1:
    v = 5 - 2.5 tau (zero at tau = 2, frame 48), x = 5 + 5 tau - 1.25 tau^2, so
    x(3) = 5 + 10 - 5 = 10. For t >= 3 the car stays at x = 10.

    Args:
        t: Time in seconds.

    Returns:
        ``(x, y)`` in meters.
    """
    if t < 1.0:
        return 5.0 * t, 0.0
    if t < 3.0:
        tau = t - 1.0
        return 5.0 + 5.0 * tau - 1.25 * tau * tau, 0.0
    return 10.0, 0.0


def without_frame(raw: list[Row], idx: int) -> list[Row]:
    """Copy rows, removing every row of frame ``idx``.

    Args:
        raw: Source rows.
        idx: Frame index to drop.

    Returns:
        The remaining rows (deep copies).
    """
    return [copy.deepcopy(r) for r in raw if r["frame_idx"] != idx]


def with_jump(raw: list[Row], first_idx: int, dx: float) -> list[Row]:
    """Copy rows, adding ``dx`` to x for every frame at or after ``first_idx``.

    Args:
        raw: Source rows.
        first_idx: First shifted frame.
        dx: Shift in meters.

    Returns:
        The modified rows (deep copies).
    """
    out = copy.deepcopy(raw)
    for row in out:
        if row["frame_idx"] >= first_idx:
            row["x"] = row["x"] + dx
    return out


def with_duplicate(raw: list[Row], idx: int) -> list[Row]:
    """Copy rows, repeating the row of frame ``idx`` immediately after itself.

    Args:
        raw: Source rows.
        idx: Frame index to duplicate.

    Returns:
        The rows with one extra identical row.
    """
    out: list[Row] = []
    for row in raw:
        out.append(copy.deepcopy(row))
        if row["frame_idx"] == idx:
            out.append(copy.deepcopy(row))
    return out


def with_field(raw: list[Row], idx: int, key: str, value: float) -> list[Row]:
    """Copy rows, overriding one field of the row for frame ``idx``.

    Args:
        raw: Source rows.
        idx: Frame index to modify.
        key: Field name.
        value: New value.

    Returns:
        The modified rows (deep copies).
    """
    out = copy.deepcopy(raw)
    for row in out:
        if row["frame_idx"] == idx:
            row[key] = value
    return out


def speed(frame_idx: int, value: float, tol: float) -> dict[str, Any]:
    """Build a speed check entry.

    Args:
        frame_idx: Frame to check.
        value: Expected speed in m/s.
        tol: Absolute tolerance.

    Returns:
        The ``speed_checks`` entry.
    """
    return {"frame_idx": frame_idx, "speed": value, "tol": tol}


def event(kind: str, start: int, end: int) -> dict[str, Any]:
    """Build an expected event entry with the standard frame tolerance.

    Args:
        kind: Event type name.
        start: Expected start frame.
        end: Expected end frame.

    Returns:
        The ``events`` entry.
    """
    return {"event_type": kind, "start_frame": start, "end_frame": end, "frame_tol": FRAME_TOL}


def case(name: str, text: str, raw: list[Row], expected: dict[str, Any]) -> dict[str, Any]:
    """Assemble one fixture document.

    Args:
        name: Case name (file stem).
        text: Description of the scenario.
        raw: Raw rows.
        expected: Expected results.

    Returns:
        The JSON-serializable fixture.
    """
    return {"case": name, "description": text, "raw": raw, "expected": expected}


def g01_raw() -> list[Row]:
    """Build the g01 rows (60 frames at 10 m/s), the base of g05 to g09.

    Returns:
        The raw rows.
    """
    return sample(60, constant_speed)


def g02_raw() -> list[Row]:
    """Build the g02 rows (160 frames), the base of g10.

    Returns:
        The raw rows.
    """
    return sample(160, hard_brake)


def g01_constant_speed() -> dict[str, Any]:
    """Build g01: 60 frames at 10 m/s along +x.

    Returns:
        The fixture document.
    """
    dist = 10.0 * 59 * DT  # path length = v * (last - first) * DT = 36.875
    expected = {
        "n_frames_out": 60,
        "error_codes": {},
        "events": [],
        "speed_checks": [speed(i, 10.0, 0.05) for i in (5, 30, 59)],
        "yaw_rate_checks": [{"frame_idx": 30, "yaw_rate": 0.0, "tol": 0.01}],
        "distance_m": dist,
        "distance_tol": 0.1,
    }
    return case("g01_constant_speed", "60 frames at 10 m/s along +x.", g01_raw(), expected)


def g02_hard_brake() -> dict[str, Any]:
    """Build g02: cruise, a -5 m/s^2 brake over frames 64..96, then a slower cruise.

    Returns:
        The fixture document.
    """
    expected = {
        "n_frames_out": 160,
        "error_codes": {},
        "events": [event("HARD_BRAKE", 64, 96)],
        "speed_checks": [speed(40, 15.0, 0.1), speed(150, 5.0, 0.1)],
    }
    text = "15 m/s cruise, -5 m/s^2 brake over frames 64..96, then 5 m/s."
    return case("g02_hard_brake", text, g02_raw(), expected)


def g03_sharp_turn() -> dict[str, Any]:
    """Build g03: straight, a +0.5 rad/s arc of radius 10 over frames 32..80, straight.

    Returns:
        The fixture document.
    """
    expected = {
        "n_frames_out": 128,
        "error_codes": {},
        "events": [event("SHARP_TURN", 32, 80)],
        "speed_checks": [speed(i, 5.0, 0.1) for i in (10, 56, 120)],
        "yaw_rate_checks": [{"frame_idx": 56, "yaw_rate": 0.5, "tol": 0.05}],
    }
    text = "5 m/s; straight, +0.5 rad/s arc of radius 10 over frames 32..80, straight at 1.5 rad."
    return case("g03_sharp_turn", text, sample(128, sharp_turn), expected)


def g04_stop() -> dict[str, Any]:
    """Build g04: cruise, a -2.5 m/s^2 deceleration to rest over frames 16..48, stopped.

    Returns:
        The fixture document.
    """
    expected = {
        "n_frames_out": 96,
        "error_codes": {},
        "events": [event("STOP", 48, 95)],
        "speed_checks": [speed(90, 0.0, 0.1)],
    }
    text = "5 m/s, -2.5 m/s^2 over frames 16..48 to rest, stopped to frame 95."
    return case("g04_stop", text, sample(96, stop), expected)


def g05_dropped_frame() -> dict[str, Any]:
    """Build g05: g01 with frame 30 omitted.

    Frame 31 follows the gap; VL-001 frames are not suspect, so its speed uses the real
    ``ts`` difference and stays 10 m/s.

    Returns:
        The fixture document.
    """
    expected = {
        "n_frames_out": 59,
        "error_codes": {"31": "VL-001"},
        "events": [],
        "speed_checks": [speed(i, 10.0, 0.2) for i in (20, 31, 45)],
    }
    raw = without_frame(g01_raw(), 30)
    return case("g05_dropped_frame", "g01 with frame 30 omitted.", raw, expected)


def g06_position_jump() -> dict[str, Any]:
    """Build g06: g01 with x increased by 5.0 m from frame 30 on.

    Returns:
        The fixture document.
    """
    expected = {
        "n_frames_out": 60,
        "error_codes": {"30": "VL-002"},
        "events": [],
        "null_kinematics_frames": [30],
        "speed_checks": [speed(i, 10.0, 0.3) for i in (20, 31, 45)],
    }
    text = (
        "g01 with x increased by 5.0 m from frame 30 on. The speed check at frame 31 means the "
        "shifted frame-30 position must be the baseline even though frame 30 is excluded from "
        "kinematics (using frame 29 would give 50 m/s)."
    )
    return case("g06_position_jump", text, with_jump(g01_raw(), 30, 5.0), expected)


def g07_duplicate_frame() -> dict[str, Any]:
    """Build g07: g01 with the row for frame 20 present twice.

    Returns:
        The fixture document.
    """
    expected = {
        "n_frames_out": 60,
        "error_codes": {"20": "VL-004"},
        "events": [],
        "null_kinematics_frames": [20],
        "speed_checks": [speed(10, 10.0, 0.3), speed(45, 10.0, 0.3)],
    }
    text = "g01 with the row for frame 20 present twice."
    return case("g07_duplicate_frame", text, with_duplicate(g01_raw(), 20), expected)


def g08_ts_backwards() -> dict[str, Any]:
    """Build g08: g01 where frame 25 has ts = ts(24) - 0.001.

    Returns:
        The fixture document.
    """
    raw = g01_raw()
    ts_bad = raw[24]["ts"] - 0.001  # ts(25) goes backwards relative to ts(24)
    expected = {
        "n_frames_out": 60,
        "error_codes": {"25": "VL-004"},
        "events": [],
        "null_kinematics_frames": [25],
        "speed_checks": [speed(20, 10.0, 0.3), speed(40, 10.0, 0.3)],
    }
    text = "g01 where frame 25 has ts = ts(24) - 0.001."
    return case("g08_ts_backwards", text, with_field(raw, 25, "ts", ts_bad), expected)


def g09_sparse_cloud() -> dict[str, Any]:
    """Build g09: g01 where frame 25 has only 10 points.

    Returns:
        The fixture document.
    """
    expected = {
        "n_frames_out": 60,
        "error_codes": {"25": "VL-006"},
        "events": [],
        "null_kinematics_frames": [25],
        "speed_checks": [speed(20, 10.0, 0.3), speed(45, 10.0, 0.3)],
    }
    text = "g01 where frame 25 has only 10 points."
    return case("g09_sparse_cloud", text, with_field(g01_raw(), 25, "n_points", 10), expected)


def g10_gap_inside_brake() -> dict[str, Any]:
    """Build g10: g02 with frame 80 omitted, in the middle of the brake.

    Returns:
        The fixture document.
    """
    expected = {
        "n_frames_out": 159,
        "error_codes": {"81": "VL-001"},
        "events": [event("HARD_BRAKE", 64, 79), event("HARD_BRAKE", 82, 96)],
        "no_event_contains_frames": [81],
    }
    text = "g02 with frame 80 omitted, in the middle of the brake."
    return case("g10_gap_inside_brake", text, without_frame(g02_raw(), 80), expected)


def g11_single_frame() -> dict[str, Any]:
    """Build g11: a single frame at the origin.

    Returns:
        The fixture document.
    """
    expected = {
        "n_frames_out": 1,
        "error_codes": {},
        "events": [],
        "distance_m": 0.0,
        "distance_tol": 1e-9,
    }
    return case("g11_single_frame", "A single frame at the origin.", [frame_row(0, 0.0)], expected)


def g12_empty() -> dict[str, Any]:
    """Build g12: no rows at all.

    Returns:
        The fixture document.
    """
    expected = {"n_frames_out": 0, "error_codes": {}, "events": []}
    return case("g12_empty", "No rows at all.", [], expected)


CASE_BUILDERS: tuple[Callable[[], dict[str, Any]], ...] = (
    g01_constant_speed,
    g02_hard_brake,
    g03_sharp_turn,
    g04_stop,
    g05_dropped_frame,
    g06_position_jump,
    g07_duplicate_frame,
    g08_ts_backwards,
    g09_sparse_cloud,
    g10_gap_inside_brake,
    g11_single_frame,
    g12_empty,
)


def build_all() -> dict[str, dict[str, Any]]:
    """Build every golden fixture in memory.

    Returns:
        Fixtures keyed by case name, in case order.
    """
    docs = [builder() for builder in CASE_BUILDERS]
    return {doc["case"]: doc for doc in sorted(docs, key=lambda d: d["case"])}


def write_all(out_dir: Path = GOLDEN_DIR) -> None:
    """Write every fixture as ``<case>.json`` (indent 2, trailing newline).

    Args:
        out_dir: Destination directory; created if missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, doc in build_all().items():
        (out_dir / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")


if __name__ == "__main__":
    write_all()
