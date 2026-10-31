"""Review round 1, items 5 and 2: input contracts of derive_frames and realism, cfg use."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fixtures.loader import load_golden, raw_dataframe
from pipeline_helpers import frames_table, run_pattern, with_codes
from vl.config import DEFAULT
from vl.etl.features import derive_frames
from vl.schemas import FRAMES_DTYPES
from vl.validation import (
    CHECK_ACCEL_P99,
    CHECK_FLAG_RATE,
    CHECK_MAX_SPEED,
    CHECK_YAW_RATE_P99,
    validate_frames,
    validate_realism,
)

PHYSICS_CHECKS = [CHECK_MAX_SPEED, CHECK_ACCEL_P99, CHECK_YAW_RATE_P99]


def g01() -> pd.DataFrame:
    """Return the constant-speed golden raw frames."""
    return raw_dataframe(load_golden("g01_constant_speed"))


def failed(report: object) -> dict[str, str]:
    """Map the failed check names of a report to their details."""
    return {c.name: c.detail for c in report.checks if not c.passed}  # type: ignore[attr-defined]


# --- derive_frames: one drive only -----------------------------------------------------------


def test_more_than_one_drive_raises_naming_the_ids() -> None:
    raw = pd.concat([g01(), g01().assign(drive_id="other")], ignore_index=True)
    with pytest.raises(ValueError, match=r"golden.*other|other.*golden"):
        derive_frames(raw)
    with_null_id = g01()
    with_null_id.loc[3, "drive_id"] = None
    with_null_id.loc[4, "drive_id"] = "second"
    with pytest.raises(ValueError, match="second"):  # a null id next to a second id
        derive_frames(with_null_id)


# --- derive_frames: dtype coercion -----------------------------------------------------------


def test_object_and_nullable_dtype_inputs_are_coerced() -> None:
    inputs = {
        "object": g01().astype(
            {"frame_idx": object, "ts": object, "n_points": object, "x": object, "y": object}
        ),
        "nullable": g01().astype(
            {"frame_idx": "Int64", "n_points": "Int64", "ts": "Float64", "x": "Float64"}
        ),
    }
    for case, raw in inputs.items():
        frames = derive_frames(raw)
        assert {n: str(t) for n, t in frames.dtypes.items()} == FRAMES_DTYPES, case
        assert frames["error_code"].isna().all(), case
        assert frames["speed"].iloc[30] == pytest.approx(10.0, abs=0.05), case


def test_nullable_input_with_a_null_centroid_is_an_empty_cloud() -> None:
    raw = g01().astype({"n_points": "Int64", "x": "Float64", "y": "Float64", "z": "Float64"})
    raw.loc[25, ["n_points", "x", "y", "z"]] = [0, pd.NA, pd.NA, pd.NA]
    frames = derive_frames(raw)
    assert frames["error_code"].dropna().tolist() == ["VL-006"]
    assert {n: str(t) for n, t in frames.dtypes.items()} == FRAMES_DTYPES


# --- validate_realism: missing or all-null columns -------------------------------------------


def test_realism_fails_a_named_check_when_a_column_is_missing_or_all_null() -> None:
    for column in ("speed", "accel", "yaw_rate"):
        variants = {
            "missing": (frames_table(200).drop(columns=[column]), "missing"),
            "all null": (frames_table(200, **{column: np.nan}), "no non-null"),
        }
        for variant, (frames, phrase) in variants.items():
            report = validate_realism(frames, "city", "car")
            names = failed(report)
            assert not report.passed, f"{column} {variant}"
            assert any(column in d and phrase in d for d in names.values()), f"{column} {variant}"
            assert set(names) <= set(PHYSICS_CHECKS), f"{column} {variant}"


def test_realism_unknown_profile_still_says_unknown_profile_for_empty_columns() -> None:
    report = validate_realism(frames_table(200).drop(columns=["speed"]), "nope", "car")
    assert "unknown profile" in failed(report)[CHECK_MAX_SPEED]


# --- cfg is used by the validators and the yaw floor -----------------------------------------


def test_validate_frames_flag_rate_comes_from_the_config_unless_overridden() -> None:
    frames = with_codes(frames_table(100), list(range(10)), "VL-001")  # 10% flagged
    assert not validate_frames(frames).passed
    assert validate_frames(frames, replace(DEFAULT, max_flag_rate=0.2)).passed
    lenient = replace(DEFAULT, max_flag_rate=0.5)
    assert not validate_frames(frames, lenient, max_flag_rate=0.01).passed
    assert validate_frames(frames, max_flag_rate=0.2).passed


def test_realism_flag_rate_and_percentile_follow_the_config() -> None:
    flagged = with_codes(frames_table(200), list(range(0, 200, 10)), "VL-001")  # 10%
    assert CHECK_FLAG_RATE in failed(validate_realism(flagged, "city", "car"))
    assert validate_realism(flagged, "city", "car", replace(DEFAULT, max_flag_rate=0.2)).passed
    accel = np.zeros(1000)
    accel[:5] = 50.0  # 0.5% of the frames: invisible at p99, visible at p99.9
    frames = frames_table(1000, accel=accel)
    assert validate_realism(frames, "city", "car").passed
    strict = replace(DEFAULT, realism_percentile=0.999)
    assert CHECK_ACCEL_P99 in failed(validate_realism(frames, "city", "car", strict))


def raw_straight(speed: float, n: int = 60) -> pd.DataFrame:
    """Return raw frames moving along x at ``speed`` with a 1-degree heading wobble."""
    raw = g01()
    ts = raw["ts"].to_numpy()
    raw["x"] = speed * ts
    raw["y"] = 0.002 * np.sin(np.arange(len(raw)) * 1.7) * run_pattern(len(raw), 0, len(raw), 1.0)
    return raw.iloc[:n]


def test_yaw_floor_follows_the_config() -> None:
    # At 0.3 m/s the heading is noise: the default floor (0.5) zeroes the yaw rate exactly.
    slow = raw_straight(0.3)
    default_yaw = derive_frames(slow)["yaw_rate"].dropna()
    assert (default_yaw == 0.0).all()
    no_floor = derive_frames(slow, replace(DEFAULT, yaw_min_speed_ms=0.0))["yaw_rate"].dropna()
    assert (no_floor != 0.0).any()
