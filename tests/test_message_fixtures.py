"""Integrity tests for the Phase 2 message fixtures (mock data only, no ``vl`` code).

The message contract is re-implemented here in plain Python, independently of any code
under ``src/``. Every valid line must satisfy it, every malformed line must violate it,
and the expected centroids are recomputed from the points with exact rational arithmetic.
"""

import filecmp
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, NoReturn

from fixtures import build_messages
from fixtures.message_loader import (
    MESSAGES_DIR,
    load_malformed_cases,
    load_valid_cases,
    malformed_lines,
    valid_lines,
)

VALID_NAMES = [
    "v01_two_points",
    "v02_negative_coordinates",
    "v03_lattice_200_points",
    "v04_with_metadata",
    "v05_without_metadata",
    "v06_integer_ts",
    "v07_extra_unknown_field",
    "v08_empty_cloud",
    "v09_large_integer_coordinate",
    "v10_float_ts",
]
MALFORMED_NAMES = [
    "m01_not_json",
    "m02_empty_line",
    "m03_json_array",
    "m04_missing_drive_id",
    "m05_missing_frame_idx",
    "m06_missing_ts",
    "m07_missing_points",
    "m08_null_points",
    "m09_points_string",
    "m10_points_flat_ints",
    "m11_inner_len_2",
    "m12_inner_len_4",
    "m13_float_coordinate",
    "m14_string_coordinate",
    "m15_frame_idx_float",
    "m16_frame_idx_string",
    "m17_ts_string",
    "m18_drive_id_number",
    "m19_drive_id_null",
    "m20_bool_coordinate",
    "m21_truncated_json",
    "m22_frame_idx_float_zero_fraction",
]
_DRIVE = "drive_id must be a string"
_FRAME = "frame_idx must be an integer"
_TS = "ts must be a number"
_POINTS = "points must be an array of [int, int, int]"
EXPECTED_PROBLEMS = {
    "m01_not_json": ["not parseable JSON"],
    "m02_empty_line": ["not parseable JSON"],
    "m03_json_array": ["not a JSON object"],
    "m04_missing_drive_id": [_DRIVE],
    "m05_missing_frame_idx": [_FRAME],
    "m06_missing_ts": [_TS],
    "m07_missing_points": [_POINTS],
    "m08_null_points": [_POINTS],
    "m09_points_string": [_POINTS],
    "m10_points_flat_ints": [_POINTS],
    "m11_inner_len_2": [_POINTS],
    "m12_inner_len_4": [_POINTS],
    "m13_float_coordinate": [_POINTS],
    "m14_string_coordinate": [_POINTS],
    "m15_frame_idx_float": [_FRAME],
    "m16_frame_idx_string": [_FRAME],
    "m17_ts_string": [_TS],
    "m18_drive_id_number": [_DRIVE],
    "m19_drive_id_null": [_DRIVE],
    "m20_bool_coordinate": [_POINTS],
    "m21_truncated_json": ["not parseable JSON"],
    "m22_frame_idx_float_zero_fraction": [_FRAME],
}
EXPECTED_VALID_KEYS = {
    "line",
    "drive_id",
    "vehicle_id",
    "vehicle_kind",
    "drive_start_utc",
    "frame_idx",
    "ts",
    "n_points",
    "x",
    "y",
    "z",
}
METADATA = ("vehicle_id", "vehicle_kind", "drive_start_utc")


def _reject_constant(name: str) -> NoReturn:
    """Reject the non-JSON constants ``NaN`` and ``Infinity`` while parsing.

    Args:
        name: The constant literal found by the parser.

    Raises:
        ValueError: Always.
    """
    raise ValueError(name)


def _is_int(value: object) -> bool:
    """Tell whether a parsed JSON value is an integer (booleans are not).

    Args:
        value: Parsed JSON value.

    Returns:
        True for an ``int`` that is not a ``bool``.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    """Tell whether a parsed JSON value is a number (booleans are not).

    Args:
        value: Parsed JSON value.

    Returns:
        True for an ``int`` or ``float`` that is not a ``bool``.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_point(value: object) -> bool:
    """Tell whether a value is an array of exactly three JSON integers.

    Args:
        value: Parsed JSON value.

    Returns:
        True for a three-integer list.
    """
    return isinstance(value, list) and len(value) == 3 and all(_is_int(c) for c in value)


def _object_problems(msg: dict[str, Any]) -> list[str]:
    """List the contract violations of a parsed JSON object.

    Args:
        msg: The parsed message object.

    Returns:
        Human-readable violations; empty when the object satisfies the contract.
    """
    problems: list[str] = []
    if not isinstance(msg.get("drive_id"), str):
        problems.append(_DRIVE)
    if not _is_int(msg.get("frame_idx")):
        problems.append(_FRAME)
    if not _is_number(msg.get("ts")):
        problems.append(_TS)
    points = msg.get("points")
    if not isinstance(points, list) or not all(_is_point(p) for p in points):
        problems.append(_POINTS)
    for key in METADATA:
        if key in msg and not isinstance(msg[key], str):
            problems.append(f"{key} must be a string when present")
    return problems


def contract_problems(line: str) -> list[str]:
    """Reference implementation of the message contract, on one raw line.

    Args:
        line: One raw line of text.

    Returns:
        Human-readable violations; empty when the line is a valid message.
    """
    try:
        msg = json.loads(line, parse_constant=_reject_constant)
    except ValueError:
        return ["not parseable JSON"]
    if not isinstance(msg, dict):
        return ["not a JSON object"]
    return _object_problems(msg)


def _centroid(points: list[list[int]]) -> tuple[Fraction, Fraction, Fraction]:
    """Exact centroid in meters of integer-centimeter points.

    Args:
        points: Non-empty list of ``[x, y, z]`` integer centimeters.

    Returns:
        Exact ``(x, y, z)`` in meters.
    """
    n = len(points)
    x, y, z = (Fraction(sum(p[axis] for p in points), n * 100) for axis in range(3))
    return x, y, z


def _expected_json() -> dict[str, Any]:
    """Read ``expected.json`` independently of the loader.

    Returns:
        The parsed expectations.
    """
    doc: dict[str, Any] = json.loads((MESSAGES_DIR / "expected.json").read_text())
    return doc


def _check_expected_layout(doc: dict[str, Any]) -> None:
    """Check expected.json has the documented layout and the same case names as the builder.

    Args:
        doc: The parsed expected.json.
    """
    path = MESSAGES_DIR / "expected.json"
    assert list(doc) == ["malformed", "valid"]
    assert path.read_text() == json.dumps(doc, indent=2, sort_keys=True) + "\n"
    assert set(doc["valid"]) == set(VALID_NAMES) and set(doc["malformed"]) == set(MALFORMED_NAMES)
    assert all(set(f) == EXPECTED_VALID_KEYS for f in doc["valid"].values())
    assert all(set(f) == {"line", "reason"} for f in doc["malformed"].values())
    built_valid = [c.case for c in build_messages.valid_specs()]
    built_bad = [c.case for c in build_messages.malformed_specs()]
    assert built_valid == VALID_NAMES and built_bad == MALFORMED_NAMES
    assert list(load_valid_cases()) == sorted(VALID_NAMES)
    assert list(load_malformed_cases()) == sorted(MALFORMED_NAMES)


def _check_valid_lines(expected: dict[str, Any]) -> None:
    """Every valid line satisfies the contract, sits on its 1-based line and matches expected.

    Args:
        expected: The ``valid`` section of expected.json.
    """
    lines = valid_lines()
    cases = load_valid_cases()
    assert len(lines) == len(VALID_NAMES)
    assert sorted(f["line"] for f in expected.values()) == list(range(1, 11))
    for name, fields in expected.items():
        assert contract_problems(cases[name].text) == [], name
        msg = json.loads(lines[fields["line"] - 1])
        assert (msg["drive_id"], msg["frame_idx"], msg["ts"]) == (
            fields["drive_id"],
            fields["frame_idx"],
            fields["ts"],
        ), name
        assert len(msg["points"]) == fields["n_points"], name
        for key in METADATA:
            assert msg.get(key) == fields[key], (name, key)


def _check_malformed_lines(expected: dict[str, Any]) -> None:
    """Every malformed line violates the contract for exactly its own mapped reason.

    Args:
        expected: The ``malformed`` section of expected.json.
    """
    lines = malformed_lines()
    cases = load_malformed_cases()
    assert list(EXPECTED_PROBLEMS) == MALFORMED_NAMES and len(lines) == len(MALFORMED_NAMES)
    for position, name in enumerate(MALFORMED_NAMES, start=1):
        assert expected[name]["line"] == position, name
        assert cases[name].text == lines[position - 1], name
        assert contract_problems(cases[name].text) == EXPECTED_PROBLEMS[name], name
        assert cases[name].reason.strip() != "", name


def test_every_message_line_is_classified_by_the_independent_contract() -> None:
    """Valid lines pass and malformed lines fail the plain-Python contract, each for its reason."""
    doc = _expected_json()
    _check_expected_layout(doc)
    _check_valid_lines(doc["valid"])
    _check_malformed_lines(doc["malformed"])


def test_expected_centroids_are_recomputed_from_the_points() -> None:
    """Centroids, in meters, equal exact rational means of the points (null if empty)."""
    expected = _expected_json()["valid"]
    lines = valid_lines()
    for name, fields in expected.items():
        points = json.loads(lines[fields["line"] - 1])["points"]
        if not points:
            assert (fields["x"], fields["y"], fields["z"]) == (None, None, None), name
            continue
        exact = _centroid(points)
        assert (fields["x"], fields["y"], fields["z"]) == tuple(float(v) for v in exact), name


def test_builder_is_deterministic_and_matches_committed(tmp_path: Path) -> None:
    """Building twice is byte-identical and equals the committed fixture files."""
    one, two = tmp_path / "one", tmp_path / "two"
    build_messages.write_all(one)
    build_messages.write_all(two)
    names = ["expected.json", "malformed_messages.jsonl", "valid_messages.jsonl"]
    assert sorted(p.name for p in MESSAGES_DIR.iterdir()) == names
    for name in names:
        assert filecmp.cmp(one / name, two / name, shallow=False), name
        assert filecmp.cmp(one / name, MESSAGES_DIR / name, shallow=False), name
