"""Tests for vl.config."""

from dataclasses import fields

import pytest

from vl.config import _RULES, DEFAULT, Config, load_config, validate_config

# (variable, rejected value, regex the error message must match: names field and rule)
INVALID_VALUES = [
    ("VL_MAX_JUMP_M", "nan", "max_jump_m"),
    ("VL_MAX_ACCEL_MS2", "inf", "max_accel_ms2"),
    ("VL_FRAME_PERIOD_S", "-0.1", "frame_period_s"),
    ("VL_FRAME_PERIOD_S", "0", r"frame_period_s.*must be > 0"),
    ("VL_HARD_BRAKE_MS2", "3.5", "hard_brake_ms2"),
    ("VL_SMOOTH_FRAMES", "0", "smooth_frames"),
    ("VL_YAW_MIN_SPEED_MS", "-0.1", "yaw_min_speed_ms"),
    ("VL_MAX_FLAG_RATE", "0", "max_flag_rate"),
    ("VL_MAX_FLAG_RATE", "1", "max_flag_rate"),
    ("VL_MAX_FLAG_RATE", "-0.5", "max_flag_rate"),
    ("VL_MAX_FLAG_RATE", "2", r"max_flag_rate.*\(0, 1\)"),
    ("VL_MAX_FALSE_POSITIVE_RATE", "0", "max_false_positive_rate"),
    ("VL_MAX_FALSE_POSITIVE_RATE", "1.5", "max_false_positive_rate"),
    ("VL_REALISM_PERCENTILE", "0", "realism_percentile"),
    ("VL_REALISM_PERCENTILE", "1", "realism_percentile"),
    ("VL_REALISM_PERCENTILE", "nan", "realism_percentile"),
]

# (variable, field, env text, expected value): coercion follows the field's type
OVERRIDES = [
    ("VL_MAX_JUMP_M", "max_jump_m", "5", 5.0),
    ("VL_MIN_CLOUD_POINTS", "min_cloud_points", "60", 60),
    ("VL_YAW_MIN_SPEED_MS", "yaw_min_speed_ms", "1.5", 1.5),
    ("VL_YAW_MIN_SPEED_MS", "yaw_min_speed_ms", "0", 0.0),  # zero floor is allowed
    ("VL_MAX_FLAG_RATE", "max_flag_rate", "0.1", 0.1),
    ("VL_MAX_FALSE_POSITIVE_RATE", "max_false_positive_rate", "0.02", 0.02),
    ("VL_REALISM_PERCENTILE", "realism_percentile", "0.95", 0.95),
]


def test_defaults_are_valid_and_unset_or_unrelated_env_changes_nothing() -> None:
    """The defaults pass validation unchanged; no VL_ variable, or an unrelated one, keeps them."""
    assert validate_config(DEFAULT) is DEFAULT
    assert load_config({}) == DEFAULT
    assert load_config({"VL_NOT_A_FIELD": "1"}) == DEFAULT


def test_env_override_is_coerced_to_the_field_type() -> None:
    """A VL_<FIELD> env var overrides the default, coerced to the field's type."""
    for variable, field, text, want in OVERRIDES:
        got = getattr(load_config({variable: text}), field)
        assert got == want and type(got) is type(want), (variable, text, got)


def test_bad_value_names_variable() -> None:
    """A malformed value raises ValueError naming the offending variable."""
    with pytest.raises(ValueError, match="VL_MAX_JUMP_M"):
        load_config({"VL_MAX_JUMP_M": "abc"})


def test_every_validation_rule_rejects_and_names_the_field() -> None:
    """Each rule (finite, positive, negative threshold, rate and percentile in (0, 1)) fires."""
    for variable, value, message in INVALID_VALUES:
        with pytest.raises(ValueError, match=message):
            load_config({variable: value})


def test_no_drop_gap_factor() -> None:
    """Dropped frames are detected by frame_idx gaps only, so no gap factor exists."""
    assert "drop_gap_factor" not in {f.name for f in fields(Config)}
    assert load_config({"VL_DROP_GAP_FACTOR": "2"}) == DEFAULT


def test_moved_thresholds_defaults() -> None:
    """The four thresholds moved into Config default to the values the code used as constants."""
    assert DEFAULT.yaw_min_speed_ms == 0.5
    assert DEFAULT.max_flag_rate == 0.05
    assert DEFAULT.max_false_positive_rate == 0.01
    assert DEFAULT.realism_percentile == 0.99


def test_every_config_field_has_exactly_one_validation_rule() -> None:
    """Adding a Config field without a `_RULES` entry (or the reverse) fails here.

    No field is intentionally outside `_RULES`: the finite check in `validate_config` is
    an extra guard over the float fields, not a replacement for the per-field rule, so the
    rule fields must equal the Config fields exactly, each listed once.
    """
    names = [name for name, _, _ in _RULES]
    assert len(names) == len(set(names))
    assert set(names) == {f.name for f in fields(Config)}
