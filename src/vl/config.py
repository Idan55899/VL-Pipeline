"""Central configuration: every pipeline threshold lives here.

Defaults are the values from the design spec; any field can be overridden with a
``VL_<FIELD>`` environment variable (for example ``VL_MAX_JUMP_M=5``).
"""

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, get_type_hints

ENV_PREFIX = "VL_"


@dataclass(frozen=True)
class Config:
    """Immutable pipeline thresholds.

    Attributes:
        frame_period_s: Nominal time between frames in seconds (16 fps).
        max_jump_m: Largest plausible displacement between consecutive frames, in meters.
        max_accel_ms2: Largest plausible absolute acceleration, in m/s^2.
        min_cloud_points: Fewer points than this marks the cloud as sparse.
        smooth_frames: Window size, in frames, for smoothing speed and heading.
        hard_brake_ms2: Acceleration at or below this is a hard brake (negative).
        hard_accel_ms2: Acceleration at or above this is a hard acceleration.
        sharp_turn_rad_s: Absolute yaw rate at or above this is a sharp turn.
        stop_speed_ms: Speed below this, in m/s, counts as stopped.
        stop_min_s: Minimum duration, in seconds, for a stop event.
        min_event_frames: Minimum consecutive frames for an event to be reported.
        yaw_min_speed_ms: Below this speed, in m/s, the heading is meaningless and the yaw
            rate is 0.
        max_flag_rate: Largest allowed share of flagged frames in a validation report
            (strictly between 0 and 1).
        max_false_positive_rate: Largest allowed share of flagged frames among the frames
            without a fault label (strictly between 0 and 1).
        realism_percentile: Percentile of ``abs(accel)`` and ``abs(yaw_rate)`` the realism
            checks compare with the profile limits (strictly between 0 and 1).
    """

    frame_period_s: float = 0.0625
    max_jump_m: float = 3.0
    max_accel_ms2: float = 15.0
    min_cloud_points: int = 50
    smooth_frames: int = 5
    hard_brake_ms2: float = -3.5
    hard_accel_ms2: float = 3.0
    sharp_turn_rad_s: float = 0.35
    stop_speed_ms: float = 0.2
    stop_min_s: float = 2.0
    min_event_frames: int = 4
    yaw_min_speed_ms: float = 0.5
    max_flag_rate: float = 0.05
    max_false_positive_rate: float = 0.01
    realism_percentile: float = 0.99


DEFAULT = Config()


def _convert(raw: str, target: Callable[[str], int | float], name: str) -> int | float:
    """Convert a raw env string to ``target``, naming the variable on failure.

    Args:
        raw: The raw string value from the environment.
        target: The field type to convert to (``int`` or ``float``).
        name: The environment variable name, used in the error message.

    Returns:
        The converted value.

    Raises:
        ValueError: If ``raw`` cannot be converted; the message names ``name``.
    """
    try:
        return target(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid {target.__name__}: {exc}") from exc


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """Collect ``VL_<FIELD>`` values from ``env``, coerced to each field's type.

    Args:
        env: Environment-like mapping of variable names to string values.

    Returns:
        Mapping of field name to coerced value, only for variables that are set.

    Raises:
        ValueError: If a value cannot be converted to the field's type.
    """
    types = get_type_hints(Config)
    overrides: dict[str, Any] = {}
    for field in fields(Config):
        name = ENV_PREFIX + field.name.upper()
        raw = env.get(name)
        if raw is not None:
            overrides[field.name] = _convert(raw, types[field.name], name)
    return overrides


# (field, predicate, rule text); every rule is one line so the checker stays small.
_RULES: tuple[tuple[str, Callable[[Any], bool], str], ...] = (
    ("frame_period_s", lambda v: v > 0, "must be > 0"),
    ("max_jump_m", lambda v: v > 0, "must be > 0"),
    ("max_accel_ms2", lambda v: v > 0, "must be > 0"),
    ("min_cloud_points", lambda v: v >= 0, "must be >= 0"),
    ("smooth_frames", lambda v: v >= 1, "must be >= 1"),
    ("hard_brake_ms2", lambda v: v < 0, "must be < 0"),
    ("hard_accel_ms2", lambda v: v > 0, "must be > 0"),
    ("sharp_turn_rad_s", lambda v: v > 0, "must be > 0"),
    ("stop_speed_ms", lambda v: v >= 0, "must be >= 0"),
    ("stop_min_s", lambda v: v > 0, "must be > 0"),
    ("min_event_frames", lambda v: v >= 1, "must be >= 1"),
    ("yaw_min_speed_ms", lambda v: v >= 0, "must be >= 0"),
    ("max_flag_rate", lambda v: 0 < v < 1, "must be in (0, 1)"),
    ("max_false_positive_rate", lambda v: 0 < v < 1, "must be in (0, 1)"),
    ("realism_percentile", lambda v: 0 < v < 1, "must be in (0, 1)"),
)


def _check_finite(cfg: Config) -> None:
    """Reject any float field that is NaN or infinite.

    Args:
        cfg: The config to check.

    Raises:
        ValueError: If a float field is not finite; the message names the field.
    """
    types = get_type_hints(Config)
    for field in fields(cfg):
        value = getattr(cfg, field.name)
        if types[field.name] is float and not math.isfinite(value):
            raise ValueError(f"{field.name}={value!r} must be finite")


def validate_config(cfg: Config) -> Config:
    """Check every field against its rule.

    Args:
        cfg: The config to validate.

    Returns:
        ``cfg`` itself, unchanged, when every rule holds.

    Raises:
        ValueError: If a rule is violated; the message names the field and the rule.
    """
    _check_finite(cfg)
    for name, holds, rule in _RULES:
        value = getattr(cfg, name)
        if not holds(value):
            raise ValueError(f"{name}={value!r} {rule}")
    return cfg


def load_config(env: Mapping[str, str] = os.environ) -> Config:
    """Build a validated `Config` from defaults plus ``VL_<FIELD>`` overrides.

    Args:
        env: Environment-like mapping; defaults to the process environment.

    Returns:
        A new frozen `Config` that passed `validate_config`.

    Raises:
        ValueError: If an override cannot be converted to the field's type, or the
            resulting config violates a validation rule.
    """
    return validate_config(replace(DEFAULT, **_env_overrides(env)))
