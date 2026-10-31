"""Review round 1, item 1: an empty cloud is VL-006 with a null centroid, not VL-005."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fixtures.loader import load_golden, raw_dataframe
from pipeline_helpers import frames_table, with_codes
from sim_helpers import make_frames
from vl.etl.features import derive_frames, detect_events
from vl.mock.profiles import PROFILE_LIMITS
from vl.validate import main
from vl.validation import (
    CHECK_CODE_COUNTS,
    CHECK_NO_NULLS,
    flag_position_jump,
    validate_frames,
    validate_realism,
)


def flagged_200() -> pd.DataFrame:
    """Return 200 baseline frames where frame 40 is an empty cloud flagged VL-006."""
    frames = frames_table(200)
    frames.loc[40, ["x", "y", "z"]] = np.nan
    return with_codes(frames, [40], "VL-006")


def raw_with_empty_cloud(frame_idx: int = 25) -> pd.DataFrame:
    """Return g01 (10 m/s) where one frame has no points and so a null centroid."""
    raw = raw_dataframe(load_golden("g01_constant_speed"))
    row = raw["frame_idx"] == frame_idx
    raw.loc[row, ["n_points", "x", "y", "z"]] = [0, np.nan, np.nan, np.nan]
    return raw


def test_empty_cloud_frame_is_vl006_with_null_position_and_does_not_contaminate() -> None:
    frames = derive_frames(raw_with_empty_cloud())
    row = frames[frames["frame_idx"] == 25].iloc[0]
    assert len(frames) == 60
    assert row["error_code"] == "VL-006"
    assert row[["x", "y", "z"]].isna().all()
    assert frames["error_code"].dropna().tolist() == ["VL-006"]
    assert row[["vx", "vy", "speed", "accel", "yaw_rate"]].isna().all()
    for frame_idx in (23, 24, 26, 27, 30):
        assert frames.loc[frames["frame_idx"] == frame_idx, "speed"].iloc[0] == pytest.approx(
            10.0, abs=0.3
        ), frame_idx
    assert detect_events(frames).empty


def test_position_jump_ignores_null_positions() -> None:
    x = pd.Series([0.0, 1.0, np.nan, 2.0, 12.0])
    y = pd.Series([0.0] * 5)
    result = flag_position_jump(x, y)
    assert result.dtype == bool
    assert result.tolist() == [False, False, False, False, True]


def test_validators_exempt_null_positions_of_vl006_rows_only() -> None:
    nan_position = frames_table(100)
    nan_position.loc[40, ["x", "y", "z"]] = np.nan
    flagged = with_codes(nan_position, [40], "VL-006")
    assert validate_frames(flagged).passed
    unflagged = validate_frames(nan_position)
    assert "'x'" in {c.name: c for c in unflagged.checks}[CHECK_NO_NULLS].detail
    assert "city" in PROFILE_LIMITS
    assert validate_realism(flagged_200(), "city", "car").passed

    other_code = frames_table(100)
    other_code.loc[40, "x"] = np.nan
    assert not validate_frames(with_codes(other_code, [40], "VL-001")).passed, "x under VL-001"
    null_ts = frames_table(100)
    null_ts.loc[40, "ts"] = np.nan
    assert not validate_frames(with_codes(null_ts, [40], "VL-006")).passed, "ts under VL-006"


def test_cli_counts_an_empty_cloud_as_vl006_not_malformed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    messages = make_frames(100)
    messages[50] = {**messages[50], "points": []}
    source = tmp_path / "frames.jsonl"
    source.write_text("".join(json.dumps(m) + "\n" for m in messages))
    code = main(["--source", str(source)])
    out = capsys.readouterr().out
    assert code == 0, out
    counts = next(line for line in out.splitlines() if CHECK_CODE_COUNTS in line)
    assert "VL-006=1" in counts
    assert "VL-005" not in counts
