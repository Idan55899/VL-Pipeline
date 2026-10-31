"""The validator's JSONL route rejects every line the Spark ingest rejects (VL-005)."""

import json
from pathlib import Path

import pytest

from fixtures.message_loader import load_malformed_cases
from vl.validate import _read_jsonl_source

_BASE = {"drive_id": "d", "frame_idx": 0, "ts": 0.0, "points": [[1, 2, 3]]}
_EXTRA = {
    "vehicle_kind_int": {**_BASE, "vehicle_kind": 5},
    "drive_start_list": {**_BASE, "drive_start_utc": ["x"]},
    "frame_idx_beyond_int64": {**_BASE, "frame_idx": 2**70},
}
_NAN_LINE = '{"drive_id": "d", "frame_idx": 0, "ts": 0.0, "points": [], "extra": NaN}'
CASES = {
    **{name: case.text for name, case in load_malformed_cases().items()},
    **{name: json.dumps(msg) for name, msg in _EXTRA.items()},
    "nan_in_extra_field": _NAN_LINE,
}


@pytest.mark.parametrize("line", CASES.values(), ids=CASES.keys())
def test_malformed_line_is_vl005_in_the_validator(line: str, tmp_path: Path) -> None:
    """One malformed line (blank ones included) yields no row and one malformed count."""
    path = tmp_path / "frames.jsonl"
    path.write_text(line + "\n")
    source = _read_jsonl_source(path)
    assert (len(source.raw), source.n_lines, source.n_malformed) == (0, 1, 1)
