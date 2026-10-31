"""Generate the Phase 2 mock Kafka messages and their hand-derived expectations.

Role in the pipeline: test support only. Run with
``.venv/bin/python -m tests.fixtures.build_messages`` to rewrite ``messages/*``.

The valid and malformed message lines later pin the behavior of ``parse_messages``.
Every expected centroid is closed-form arithmetic written next to its case (integer
centimeter points, mean over the points, divided by 100 to get meters); nothing here
imports or runs any code under ``src/vl``.

Import note: pytest puts ``tests/`` on ``sys.path`` so the tests import this module as
``fixtures.build_messages``, while the generator is run as ``tests.fixtures.build_messages``;
neither this module nor the loader imports the other, so both spellings work.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MESSAGES_DIR = Path(__file__).parent / "messages"
DRIVE_ID = "drive-001"

Message = dict[str, Any]
Centroid = tuple[float, float, float] | None


@dataclass(frozen=True)
class ValidSpec:
    """One valid message line with its hand-derived expectations.

    Attributes:
        case: Case name such as ``"v01_two_points"``.
        text: The raw JSONL line.
        fields: Expected fields (everything in ``expected.json`` except ``line``).
    """

    case: str
    text: str
    fields: dict[str, Any]


@dataclass(frozen=True)
class MalformedSpec:
    """One malformed raw line and the reason it must be rejected (VL-005).

    Attributes:
        case: Case name such as ``"m01_not_json"``.
        text: The raw line (not necessarily JSON).
        reason: Why the line is malformed.
    """

    case: str
    text: str
    reason: str


def dump(message: Message) -> str:
    """Serialize a message deterministically (sorted keys, compact separators).

    Args:
        message: The message object.

    Returns:
        One line of JSON text.
    """
    return json.dumps(message, sort_keys=True, separators=(",", ":"))


def message(frame_idx: int, ts: float, points: list[list[int]], **extra: object) -> Message:
    """Build a message object for the shared test drive.

    Args:
        frame_idx: Frame index.
        ts: Timestamp in seconds (an ``int`` is kept as an integer literal).
        points: Cloud points in integer centimeters.
        **extra: Additional fields (metadata or unknown fields).

    Returns:
        The message dictionary.
    """
    return {"drive_id": DRIVE_ID, "frame_idx": frame_idx, "ts": ts, "points": points, **extra}


def valid(
    case: str, msg: Message, n_points: int, centroid: Centroid, ts: float | None = None
) -> ValidSpec:
    """Pair a message with its hand-derived expectations.

    Args:
        case: Case name.
        msg: The message object.
        n_points: Hand-counted number of points.
        centroid: Hand-derived ``(x, y, z)`` in meters, or ``None`` for an empty cloud.
        ts: Expected float timestamp; defaults to the message's ``ts``.

    Returns:
        The `ValidSpec` (identity and metadata fields are copied from the message).
    """
    x, y, z = centroid if centroid is not None else (None, None, None)
    fields = {
        "drive_id": msg["drive_id"],
        "vehicle_id": msg.get("vehicle_id"),
        "vehicle_kind": msg.get("vehicle_kind"),
        "drive_start_utc": msg.get("drive_start_utc"),
        "frame_idx": msg["frame_idx"],
        "ts": msg["ts"] if ts is None else ts,
        "n_points": n_points,
        "x": x,
        "y": y,
        "z": z,
    }
    return ValidSpec(case, dump(msg), fields)


def lattice(cx: int, cy: int, cz: int) -> list[list[int]]:
    """Build the 10 x 5 x 4 lattice at 50 cm pitch centered at ``(cx, cy, cz)`` cm.

    Args:
        cx: Center x in cm.
        cy: Center y in cm.
        cz: Center z in cm.

    Returns:
        The 200 points, ordered by i, then j, then k.
    """
    return [
        [cx + 50 * i - 225, cy + 50 * j - 100, cz + 50 * k - 75]
        for i in range(10)
        for j in range(5)
        for k in range(4)
    ]


def v01_two_points() -> ValidSpec:
    """Two-point cloud.

    Derivation (cm): x = (0 + 100) / 2 = 50, y = (0 + 200) / 2 = 100,
    z = (0 + 300) / 2 = 150, so the centroid is (0.5, 1.0, 1.5) m.

    Returns:
        The case.
    """
    msg = message(0, 0.0, [[0, 0, 0], [100, 200, 300]])
    return valid("v01_two_points", msg, 2, (0.5, 1.0, 1.5))


def v02_negative_coordinates() -> ValidSpec:
    """Negative coordinates.

    Derivation (cm): x = (-100 + 100) / 2 = 0, y = (0 + 0) / 2 = 0,
    z = (50 + 150) / 2 = 100, so the centroid is (0.0, 0.0, 1.0) m.

    Returns:
        The case.
    """
    msg = message(1, 0.0625, [[-100, 0, 50], [100, 0, 150]])
    return valid("v02_negative_coordinates", msg, 2, (0.0, 0.0, 1.0))


def v03_lattice_200_points() -> ValidSpec:
    """10 x 5 x 4 lattice at 50 cm pitch centered at (1234, -567, 89) cm.

    Derivation (cm): n = 10 * 5 * 4 = 200. The offsets 50 i - 225 for i = 0..9 sum to
    50 * 45 - 2250 = 0, the offsets 50 j - 100 for j = 0..4 sum to 50 * 10 - 500 = 0,
    and the offsets 50 k - 75 for k = 0..3 sum to 50 * 6 - 300 = 0, so the mean is the
    center: (1234, -567, 89) cm = (12.34, -5.67, 0.89) m, exactly.

    Returns:
        The case.
    """
    msg = message(2, 0.125, lattice(1234, -567, 89))
    return valid("v03_lattice_200_points", msg, 200, (12.34, -5.67, 0.89))


def v04_with_metadata() -> ValidSpec:
    """All three metadata fields present, one-point cloud.

    Derivation (cm): a single point is its own mean, (10, 20, 30) cm = (0.1, 0.2, 0.3) m.

    Returns:
        The case.
    """
    msg = message(
        3,
        0.1875,
        [[10, 20, 30]],
        vehicle_id="veh-drive-001",
        vehicle_kind="car",
        drive_start_utc="2026-01-01T08:00:00+00:00",
    )
    return valid("v04_with_metadata", msg, 1, (0.1, 0.2, 0.3))


def v05_without_metadata() -> ValidSpec:
    """The v04 message without any metadata (expected metadata all null).

    Derivation: same single point as v04, so the centroid is (0.1, 0.2, 0.3) m.

    Returns:
        The case.
    """
    msg = message(3, 0.1875, [[10, 20, 30]])
    return valid("v05_without_metadata", msg, 1, (0.1, 0.2, 0.3))


def v06_integer_ts() -> ValidSpec:
    """Integer ``ts`` literal (3), expected as the float 3.0.

    Derivation (cm): (100 + 300) / 2 = 200, (-50 + 50) / 2 = 0, (0 + 0) / 2 = 0, so the
    centroid is (2.0, 0.0, 0.0) m.

    Returns:
        The case.
    """
    msg = message(4, 3, [[100, -50, 0], [300, 50, 0]])
    return valid("v06_integer_ts", msg, 2, (2.0, 0.0, 0.0), ts=3.0)


def v07_extra_unknown_field() -> ValidSpec:
    """An unknown extra field, which the contract ignores.

    Derivation (cm): a single point (-250, 0, 75) is its own mean = (-2.5, 0.0, 0.75) m.

    Returns:
        The case.
    """
    msg = message(5, 0.3125, [[-250, 0, 75]], extra={"a": 1})
    return valid("v07_extra_unknown_field", msg, 1, (-2.5, 0.0, 0.75))


def v08_empty_cloud() -> ValidSpec:
    """Empty cloud: valid, zero points, centroid null.

    Derivation: the mean of zero points is undefined, so x, y, z are null.

    Returns:
        The case.
    """
    msg = message(6, 0.375, [])
    return valid("v08_empty_cloud", msg, 0, None)


def v09_large_integer_coordinate() -> ValidSpec:
    """Two large coordinates whose sum overflows int32, pinning 64-bit centroid math.

    Derivation (cm): each x = 2000000000 < 2**31 = 2147483648, but the sum
    2000000000 + 2000000000 = 4000000000 exceeds int32. The mean is 4000000000 / 2 =
    2000000000 cm = 20000000.0 m exactly, and y = z = 0.

    Returns:
        The case.
    """
    msg = message(7, 0.4375, [[2000000000, 0, 0], [2000000000, 0, 0]])
    return valid("v09_large_integer_coordinate", msg, 2, (20000000.0, 0.0, 0.0))


def v10_float_ts() -> ValidSpec:
    """A fractional float ``ts`` (1.0625 = 17 / 16, exact in binary floating point).

    Derivation (cm): one point (5, 5, 5) is its own mean = (0.05, 0.05, 0.05) m.

    Returns:
        The case.
    """
    msg = message(8, 1.0625, [[5, 5, 5]])
    return valid("v10_float_ts", msg, 1, (0.05, 0.05, 0.05))


VALID_BUILDERS: tuple[Callable[[], ValidSpec], ...] = (
    v01_two_points,
    v02_negative_coordinates,
    v03_lattice_200_points,
    v04_with_metadata,
    v05_without_metadata,
    v06_integer_ts,
    v07_extra_unknown_field,
    v08_empty_cloud,
    v09_large_integer_coordinate,
    v10_float_ts,
)


def base() -> Message:
    """Build a good one-point message that malformed cases then break.

    Returns:
        A valid message object.
    """
    return message(0, 0.0, [[1, 2, 3]])


def without(key: str) -> str:
    """Dump the base message with one field removed.

    Args:
        key: Field to drop.

    Returns:
        One line of JSON text.
    """
    msg = base()
    del msg[key]
    return dump(msg)


def with_field(key: str, value: object) -> str:
    """Dump the base message with one field overwritten.

    Args:
        key: Field to set.
        value: New value.

    Returns:
        One line of JSON text.
    """
    return dump({**base(), key: value})


def with_points(points: object) -> str:
    """Dump the base message with its ``points`` replaced.

    Args:
        points: New ``points`` value (any JSON value).

    Returns:
        One line of JSON text.
    """
    return with_field("points", points)


def m01_not_json() -> MalformedSpec:
    """Plain text that is not JSON.

    Returns:
        The case.
    """
    return MalformedSpec("m01_not_json", "not json", "text is not JSON")


def m02_empty_line() -> MalformedSpec:
    """An empty line.

    Returns:
        The case.
    """
    return MalformedSpec("m02_empty_line", "", "empty line is not JSON")


def m03_json_array() -> MalformedSpec:
    """Valid JSON of the wrong shape: an array instead of an object.

    Returns:
        The case.
    """
    return MalformedSpec("m03_json_array", "[1, 2]", "JSON value is an array, not an object")


def m04_missing_drive_id() -> MalformedSpec:
    """No ``drive_id``.

    Returns:
        The case.
    """
    return MalformedSpec("m04_missing_drive_id", without("drive_id"), "drive_id is missing")


def m05_missing_frame_idx() -> MalformedSpec:
    """No ``frame_idx``.

    Returns:
        The case.
    """
    return MalformedSpec("m05_missing_frame_idx", without("frame_idx"), "frame_idx is missing")


def m06_missing_ts() -> MalformedSpec:
    """No ``ts``.

    Returns:
        The case.
    """
    return MalformedSpec("m06_missing_ts", without("ts"), "ts is missing")


def m07_missing_points() -> MalformedSpec:
    """No ``points``.

    Returns:
        The case.
    """
    return MalformedSpec("m07_missing_points", without("points"), "points is missing")


def m08_null_points() -> MalformedSpec:
    """``points`` is null.

    Returns:
        The case.
    """
    return MalformedSpec("m08_null_points", with_points(None), "points is null, not an array")


def m09_points_string() -> MalformedSpec:
    """``points`` is a string.

    Returns:
        The case.
    """
    return MalformedSpec("m09_points_string", with_points("garbage"), "points is a string")


def m10_points_flat_ints() -> MalformedSpec:
    """``points`` is a flat array of integers instead of an array of triples.

    Returns:
        The case.
    """
    reason = "points is a flat integer array, not an array of arrays"
    return MalformedSpec("m10_points_flat_ints", with_points([1, 2, 3]), reason)


def m11_inner_len_2() -> MalformedSpec:
    """A point with two coordinates.

    Returns:
        The case.
    """
    return MalformedSpec("m11_inner_len_2", with_points([[1, 2]]), "a point has 2 coordinates")


def m12_inner_len_4() -> MalformedSpec:
    """A point with four coordinates.

    Returns:
        The case.
    """
    reason = "a point has 4 coordinates"
    return MalformedSpec("m12_inner_len_4", with_points([[1, 2, 3, 4]]), reason)


def m13_float_coordinate() -> MalformedSpec:
    """A float coordinate (coordinates must be integers).

    Returns:
        The case.
    """
    reason = "coordinate 0.5 is a float, not an integer"
    return MalformedSpec("m13_float_coordinate", with_points([[0.5, 0, 0]]), reason)


def m14_string_coordinate() -> MalformedSpec:
    """A string coordinate.

    Returns:
        The case.
    """
    reason = "coordinate is a string, not an integer"
    return MalformedSpec("m14_string_coordinate", with_points([["1", 0, 0]]), reason)


def m15_frame_idx_float() -> MalformedSpec:
    """``frame_idx`` is a float.

    Returns:
        The case.
    """
    reason = "frame_idx 1.5 is a float, not an integer"
    return MalformedSpec("m15_frame_idx_float", with_field("frame_idx", 1.5), reason)


def m16_frame_idx_string() -> MalformedSpec:
    """``frame_idx`` is a string.

    Returns:
        The case.
    """
    reason = "frame_idx is a string, not an integer"
    return MalformedSpec("m16_frame_idx_string", with_field("frame_idx", "5"), reason)


def m17_ts_string() -> MalformedSpec:
    """``ts`` is a string.

    Returns:
        The case.
    """
    reason = "ts is a string, not a number"
    return MalformedSpec("m17_ts_string", with_field("ts", "0.0625"), reason)


def m18_drive_id_number() -> MalformedSpec:
    """``drive_id`` is a number.

    Returns:
        The case.
    """
    reason = "drive_id is a number, not a string"
    return MalformedSpec("m18_drive_id_number", with_field("drive_id", 7), reason)


def m19_drive_id_null() -> MalformedSpec:
    """``drive_id`` is null.

    Returns:
        The case.
    """
    reason = "drive_id is null, not a string"
    return MalformedSpec("m19_drive_id_null", with_field("drive_id", None), reason)


def m20_bool_coordinate() -> MalformedSpec:
    """A boolean coordinate (booleans are not integers).

    Returns:
        The case.
    """
    reason = "coordinate true is a boolean, not an integer"
    return MalformedSpec("m20_bool_coordinate", with_points([[True, 0, 0]]), reason)


def m21_truncated_json() -> MalformedSpec:
    """A valid message cut in the middle (a broken record).

    Returns:
        The case.
    """
    full = dump(base())
    return MalformedSpec("m21_truncated_json", full[: len(full) // 2], "JSON is truncated")


def m22_frame_idx_float_zero_fraction() -> MalformedSpec:
    """``frame_idx`` is the float literal 1.0 (a zero fraction is still not an integer).

    Returns:
        The case.
    """
    reason = "frame_idx 1.0 is a float literal, not an integer (strict contract)"
    return MalformedSpec("m22_frame_idx_float_zero_fraction", with_field("frame_idx", 1.0), reason)


MALFORMED_BUILDERS: tuple[Callable[[], MalformedSpec], ...] = (
    m01_not_json,
    m02_empty_line,
    m03_json_array,
    m04_missing_drive_id,
    m05_missing_frame_idx,
    m06_missing_ts,
    m07_missing_points,
    m08_null_points,
    m09_points_string,
    m10_points_flat_ints,
    m11_inner_len_2,
    m12_inner_len_4,
    m13_float_coordinate,
    m14_string_coordinate,
    m15_frame_idx_float,
    m16_frame_idx_string,
    m17_ts_string,
    m18_drive_id_number,
    m19_drive_id_null,
    m20_bool_coordinate,
    m21_truncated_json,
    m22_frame_idx_float_zero_fraction,
)


def valid_specs() -> list[ValidSpec]:
    """Build every valid case, in file order.

    Returns:
        The valid specs.
    """
    return [builder() for builder in VALID_BUILDERS]


def malformed_specs() -> list[MalformedSpec]:
    """Build every malformed case, in file order.

    Returns:
        The malformed specs.
    """
    return [builder() for builder in MALFORMED_BUILDERS]


def build_expected() -> dict[str, dict[str, dict[str, Any]]]:
    """Assemble ``expected.json``; the 1-based line is the case's position in its file.

    Returns:
        The expectations keyed by ``valid`` / ``malformed`` and case name.
    """
    good = {s.case: {"line": n, **s.fields} for n, s in enumerate(valid_specs(), start=1)}
    bad = {
        s.case: {"line": n, "reason": s.reason} for n, s in enumerate(malformed_specs(), start=1)
    }
    return {"valid": good, "malformed": bad}


def write_lines(path: Path, lines: list[str]) -> None:
    """Write raw lines, one per line, each terminated by a newline.

    Args:
        path: Destination file.
        lines: The raw lines (an empty string yields an empty line).
    """
    path.write_text("".join(f"{line}\n" for line in lines))


def write_all(out_dir: Path = MESSAGES_DIR) -> None:
    """Write both JSONL files and ``expected.json`` (indent 2, sorted keys).

    Args:
        out_dir: Destination directory; created if missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    write_lines(out_dir / "valid_messages.jsonl", [s.text for s in valid_specs()])
    write_lines(out_dir / "malformed_messages.jsonl", [s.text for s in malformed_specs()])
    expected = json.dumps(build_expected(), indent=2, sort_keys=True) + "\n"
    (out_dir / "expected.json").write_text(expected)


if __name__ == "__main__":
    write_all()
