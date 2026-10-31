"""Unit tests for the per-frame error-code rule functions of ``vl.validation``."""

from dataclasses import replace

import numpy as np
import pandas as pd

from pipeline_helpers import DT
from vl.config import DEFAULT
from vl.validation import (
    CODE_PRECEDENCE,
    assign_error_codes,
    flag_duplicate_or_backwards_ts,
    flag_frame_gaps,
    flag_impossible_accel,
    flag_position_jump,
    flag_sparse_cloud,
)


def flags(result: "pd.Series[bool]") -> list[bool]:
    """Return the boolean mask as a plain list, asserting its dtype is bool."""
    assert result.dtype == bool
    return [bool(value) for value in result]


def ts_of(frame_idx: list[int]) -> pd.Series:
    """Return clean timestamps (``idx * DT``) for a list of frame indices."""
    return pd.Series([i * DT for i in frame_idx])


# --- VL-004 -------------------------------------------------------------------------------


def test_duplicate_or_backwards_ts_flags() -> None:
    cases = [
        ("clean", [0, 1, 2, 3], ts_of([0, 1, 2, 3]), [False] * 4),
        (
            "duplicate frame_idx flags every occurrence",
            [0, 1, 1, 2],
            pd.Series([0.0, DT, DT, 2 * DT]),
            [False, True, True, False],
        ),
        (
            "backwards ts flags only the offending row",
            [0, 1, 2, 3],
            pd.Series([0.0, DT, DT - 0.001, 3 * DT]),
            [False, False, True, False],
        ),
        (
            "equal ts with distinct frame_idx",
            [0, 1, 2],
            pd.Series([0.0, DT, DT]),
            [False, False, True],
        ),
    ]
    for case, idx, ts, want in cases:
        assert flags(flag_duplicate_or_backwards_ts(pd.Series(idx), ts)) == want, case


def test_duplicate_ts_rule_keeps_the_input_index_and_handles_empty() -> None:
    idx = pd.Series([4, 5, 5], index=[10, 11, 12])
    ts = pd.Series([0.0, DT, DT], index=[10, 11, 12])
    result = flag_duplicate_or_backwards_ts(idx, ts)
    assert list(result.index) == [10, 11, 12]
    empty = flag_duplicate_or_backwards_ts(pd.Series([], dtype="int64"), pd.Series([], dtype=float))
    assert len(empty) == 0


# --- VL-006 -------------------------------------------------------------------------------


def test_sparse_cloud_boundary_is_strictly_below_min_points_and_follows_the_config() -> None:
    n_points = pd.Series([200, 50, 49, 10, 0])
    assert flags(flag_sparse_cloud(n_points)) == [False, False, True, True, True]
    cfg = replace(DEFAULT, min_cloud_points=100)
    assert flags(flag_sparse_cloud(pd.Series([150, 100, 99]), cfg)) == [False, False, True]


# --- VL-001 -------------------------------------------------------------------------------


def test_frame_gap_flags_the_first_row_after_each_gap_by_frame_idx_only() -> None:
    cases = [
        ("one gap", [0, 1, 2, 5, 6], [False, False, False, True, False]),
        (
            "one row per gap whatever its length",
            [0, 1, 7, 8, 10],
            [False, False, True, False, True],
        ),
        ("first row is never flagged", [40, 41], [False, False]),
        ("empty", [], []),
    ]
    for case, idx, want in cases:
        assert flags(flag_frame_gaps(pd.Series(idx, dtype="int64"))) == want, case


# --- VL-002 -------------------------------------------------------------------------------


def test_position_jump_threshold_is_strict_in_both_directions() -> None:
    x = pd.Series([0.0, 1.0, 2.0, 7.0, 8.0])
    assert flags(flag_position_jump(x, pd.Series([0.0] * 5))) == [False, False, False, True, False]
    for step, expected in [(3.0, False), (3.01, True), (-3.01, True)]:
        result = flag_position_jump(pd.Series([0.0, step]), pd.Series([0.0, 0.0]))
        assert flags(result) == [False, expected], step


def test_position_jump_is_planar_follows_the_config_and_handles_empty() -> None:
    # hypot(2, 2.5) = 3.2 > 3 although each axis alone is below 3.
    planar = flag_position_jump(pd.Series([0.0, 2.0]), pd.Series([0.0, 2.5]))
    assert flags(planar) == [False, True]
    cfg = replace(DEFAULT, max_jump_m=1.0)
    assert flags(flag_position_jump(pd.Series([0.0, 1.5]), pd.Series([0.0, 0.0]), cfg)) == [
        False,
        True,
    ]
    assert len(flag_position_jump(pd.Series([], dtype=float), pd.Series([], dtype=float))) == 0


# --- VL-003 -------------------------------------------------------------------------------


def test_impossible_accel_is_strict_in_both_signs_and_follows_the_config() -> None:
    accel = pd.Series([0.0, 15.0, 15.1, -15.1, -15.0, np.nan])
    assert flags(flag_impossible_accel(accel)) == [False, False, True, True, False, False]
    cfg = replace(DEFAULT, max_accel_ms2=5.0)
    assert flags(flag_impossible_accel(pd.Series([5.0, 5.1]), cfg)) == [False, True]


# --- precedence ---------------------------------------------------------------------------


def test_code_precedence_constant_matches_the_spec() -> None:
    assert CODE_PRECEDENCE == ["VL-004", "VL-006", "VL-001", "VL-002", "VL-003"]


def test_assign_error_codes_picks_the_highest_precedence_flag() -> None:
    for position in range(5):
        table = pd.DataFrame({code: [i >= position] for i, code in enumerate(CODE_PRECEDENCE)})
        assert assign_error_codes(table).tolist() == [CODE_PRECEDENCE[position]], position


def test_assign_error_codes_rows_are_independent_and_missing_columns_unflagged() -> None:
    none = pd.DataFrame({code: [False] * 3 for code in CODE_PRECEDENCE})
    assert assign_error_codes(none).isna().all()
    table = pd.DataFrame(
        {
            "VL-004": [False, False, True],
            "VL-006": [False, True, True],
            "VL-001": [True, True, True],
            "VL-002": [False, False, False],
            "VL-003": [False, False, False],
        },
        index=[7, 8, 9],
    )
    codes = assign_error_codes(table)
    assert list(codes.index) == [7, 8, 9]
    assert codes.tolist() == ["VL-001", "VL-006", "VL-004"]
    partial = assign_error_codes(pd.DataFrame({"VL-003": [True, False]}))
    assert partial.iloc[0] == "VL-003"
    assert pd.isna(partial.iloc[1])
