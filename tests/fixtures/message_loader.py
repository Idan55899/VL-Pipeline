"""Load the committed Phase 2 message fixtures as typed objects.

Role in the pipeline: test support only. The ``parse_messages`` tests read their input
lines and expected results here.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MESSAGES_DIR = Path(__file__).parent / "messages"


@dataclass(frozen=True)
class ValidCase:
    """One valid message line with its hand-derived expectations.

    Attributes:
        case: Case name, for example ``"v01_two_points"``.
        line: 1-based line number in ``valid_messages.jsonl``.
        text: The raw line.
        drive_id: Expected drive id.
        vehicle_id: Expected vehicle id, or None when absent.
        vehicle_kind: Expected vehicle kind, or None when absent.
        drive_start_utc: Expected drive start, or None when absent.
        frame_idx: Expected frame index.
        ts: Expected timestamp as a float.
        n_points: Expected number of cloud points.
        x: Expected centroid x in meters, or None for an empty cloud.
        y: Expected centroid y in meters, or None for an empty cloud.
        z: Expected centroid z in meters, or None for an empty cloud.
    """

    case: str
    line: int
    text: str
    drive_id: str
    vehicle_id: str | None
    vehicle_kind: str | None
    drive_start_utc: str | None
    frame_idx: int
    ts: float
    n_points: int
    x: float | None
    y: float | None
    z: float | None


@dataclass(frozen=True)
class MalformedCase:
    """One malformed raw line that must be rejected as VL-005.

    Attributes:
        case: Case name, for example ``"m01_not_json"``.
        line: 1-based line number in ``malformed_messages.jsonl``.
        text: The raw line (possibly empty or not JSON).
        reason: Why the line is malformed.
    """

    case: str
    line: int
    text: str
    reason: str


def _read_lines(name: str) -> list[str]:
    """Read a fixture file as raw lines, keeping empty lines.

    Args:
        name: File name inside `MESSAGES_DIR`.

    Returns:
        The lines without their trailing newline.
    """
    text = (MESSAGES_DIR / name).read_text()
    return text.removesuffix("\n").split("\n")


def valid_lines() -> list[str]:
    """Read the valid message lines.

    Returns:
        One raw line per valid message, in file order.
    """
    return _read_lines("valid_messages.jsonl")


def malformed_lines() -> list[str]:
    """Read the malformed raw lines (including the empty one).

    Returns:
        One raw line per malformed case, in file order.
    """
    return _read_lines("malformed_messages.jsonl")


def _expected() -> dict[str, dict[str, dict[str, Any]]]:
    """Read ``expected.json``.

    Returns:
        The parsed expectations.
    """
    doc: dict[str, dict[str, dict[str, Any]]] = json.loads(
        (MESSAGES_DIR / "expected.json").read_text()
    )
    return doc


def load_valid_cases() -> dict[str, ValidCase]:
    """Load the valid cases with their expectations.

    Returns:
        Cases keyed by name, sorted by name.
    """
    lines = valid_lines()
    cases: dict[str, ValidCase] = {}
    for name, fields in sorted(_expected()["valid"].items()):
        text = lines[fields["line"] - 1]
        cases[name] = ValidCase(name, text=text, **fields)
    return cases


def load_malformed_cases() -> dict[str, MalformedCase]:
    """Load the malformed cases with their reasons.

    Returns:
        Cases keyed by name, sorted by name.
    """
    lines = malformed_lines()
    cases: dict[str, MalformedCase] = {}
    for name, fields in sorted(_expected()["malformed"].items()):
        line = fields["line"]
        cases[name] = MalformedCase(name, line, lines[line - 1], fields["reason"])
    return cases
