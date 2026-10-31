"""The single validator: per-frame error-code rules and the data-quality report.

Role in the pipeline: every error-code rule is a small function here, used by
``derive_frames`` to flag frames and by the standalone report command (``vl.validate``);
no rule logic exists anywhere else. The report functions check a derived ``frames`` table
(``validate_frames``) and the physical realism of one drive (``validate_realism``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vl.config import DEFAULT, Config
from vl.errors import ErrorCode
from vl.mock.profiles import PROFILE_LIMITS, ProfileLimits
from vl.mock.vehicles import VEHICLES
from vl.schemas import FRAMES_COLUMNS, FRAMES_DTYPES

# Names of the report checks (the tests and the CLI refer to checks by these names).
CHECK_REQUIRED_COLUMNS = "required_columns"
CHECK_DTYPES = "column_dtypes"
CHECK_NO_NULLS = "no_nulls"
CHECK_FINITE = "finite_values"
CHECK_TS_ORDER = "ts_non_decreasing"
CHECK_CODE_COUNTS = "error_code_counts"
CHECK_FLAG_RATE = "flag_rate"
CHECK_KNOWN_PROFILE = "known_profile"
CHECK_KNOWN_VEHICLE_KIND = "known_vehicle_kind"
CHECK_MAX_SPEED = "max_speed"
CHECK_ACCEL_P99 = "accel_p99"
CHECK_YAW_RATE_P99 = "yaw_rate_p99"
CHECK_SOURCE_NOT_EMPTY = "source_not_empty"
CHECK_LABELED_FAULTS = "labeled_faults_detected"
CHECK_FALSE_POSITIVES = "false_positive_rate"

# Columns that must never be null in a ``frames`` table.
KEY_COLUMNS = ["drive_id", "frame_idx", "ts", "x", "y", "z"]
# Position columns: null (an empty cloud has no centroid) only on rows flagged VL-006.
POSITION_COLUMNS = ["x", "y", "z"]
# Error codes in precedence order (highest first).
CODE_PRECEDENCE = [
    ErrorCode.DUPLICATE_FRAME.value,
    ErrorCode.SPARSE_CLOUD.value,
    ErrorCode.DROPPED_FRAME.value,
    ErrorCode.POSITION_JUMP.value,
    ErrorCode.IMPOSSIBLE_ACCEL.value,
]


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one named check.

    Attributes:
        name: Check name (one of the ``CHECK_*`` constants).
        passed: Whether the check passed.
        detail: Human-readable explanation, naming offending columns or values on failure.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    """A set of check results.

    Attributes:
        checks: The results, in the order the checks ran.
    """

    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """Whether every check passed.

        Returns:
            ``True`` when all checks passed.
        """
        return all(check.passed for check in self.checks)

    def to_text(self) -> str:
        """Render the report as text, one line per check: ``<PASS|FAIL> <name>: <detail>``.

        Returns:
            The report text.
        """
        lines = [
            f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.detail}"
            for check in self.checks
        ]
        return "\n".join(lines)


# --- per-frame error-code rules -------------------------------------------------------------


def flag_duplicate_or_backwards_ts(
    frame_idx: pd.Series[int], ts: pd.Series[float]
) -> pd.Series[bool]:
    """Flag VL-004: repeated ``frame_idx`` or a ``ts`` that does not increase.

    The rows must be in ``frame_idx`` order (stable, as emitted). A row is flagged when its
    ``frame_idx`` occurs more than once (every occurrence) or when its ``ts`` is less than or
    equal to the ``ts`` of the previous row. The first row is never flagged by the ``ts`` rule.

    Known limitation: a timestamp that spikes forward is not itself flagged; the row after it
    is (its ``ts`` is below the spike's). The spike row therefore keeps a wrong time step and
    contaminates the kinematics of its neighbours.

    Args:
        frame_idx: Frame index per row.
        ts: Timestamp per row, same index as ``frame_idx``.

    Returns:
        A boolean Series with the index of ``frame_idx``.
    """
    repeated = frame_idx.duplicated(keep=False)
    not_increasing = ts <= ts.shift(1)
    return (repeated | not_increasing).astype(bool)


def flag_sparse_cloud(n_points: pd.Series[int], cfg: Config = DEFAULT) -> pd.Series[bool]:
    """Flag VL-006: a point cloud with fewer than ``cfg.min_cloud_points`` points.

    Args:
        n_points: Number of cloud points per row.
        cfg: Pipeline thresholds.

    Returns:
        A boolean Series with the index of ``n_points``; ``True`` where
        ``n_points < cfg.min_cloud_points``.
    """
    return (n_points < cfg.min_cloud_points).astype(bool)


def flag_frame_gaps(frame_idx: pd.Series[int]) -> pd.Series[bool]:
    """Flag VL-001: the first row after a gap in ``frame_idx`` (timestamps are not used).

    The input must be sorted ascending without duplicates. A row is flagged when its
    ``frame_idx`` exceeds the previous row's by more than 1 (one flag per gap, however many
    frames were lost); the first row is never flagged.

    Args:
        frame_idx: Frame index per row.

    Returns:
        A boolean Series with the index of ``frame_idx``.
    """
    return (frame_idx.diff() > 1).astype(bool)


def flag_position_jump(
    x: pd.Series[float], y: pd.Series[float], cfg: Config = DEFAULT
) -> pd.Series[bool]:
    """Flag VL-002: a row whose planar displacement from the previous row is too large.

    A row is flagged when ``hypot(dx, dy) > cfg.max_jump_m`` relative to the previous row,
    regardless of any gap in ``frame_idx``; the first row is never flagged. Rows with a null
    position (an empty cloud) are never flagged, and the next row is compared with the last
    row that has a position.

    Known limitation: the comparison is with the previous row (not with the last trusted
    one), so a sparse-cloud frame (VL-006) whose centroid is shifted by more than
    ``cfg.max_jump_m`` can raise a spurious VL-002 on the frame after it.

    Args:
        x: Centroid x in meters per row.
        y: Centroid y in meters per row, same index as ``x``.
        cfg: Pipeline thresholds.

    Returns:
        A boolean Series with the index of ``x``.
    """
    known = x.notna() & y.notna()
    step = pd.Series(np.hypot(x[known].diff(), y[known].diff()), index=x[known].index)
    step = step.reindex(x.index)
    return (step > cfg.max_jump_m).astype(bool)


def flag_impossible_accel(accel: pd.Series[float], cfg: Config = DEFAULT) -> pd.Series[bool]:
    """Flag VL-003: an acceleration beyond physical limits.

    Args:
        accel: Acceleration in m/s^2 per row; null where undefined.
        cfg: Pipeline thresholds.

    Returns:
        A boolean Series with the index of ``accel``; ``True`` where
        ``abs(accel) > cfg.max_accel_ms2`` (null is never flagged).
    """
    return (accel.abs() > cfg.max_accel_ms2).astype(bool)


def assign_error_codes(flags: pd.DataFrame) -> pd.Series[str]:
    """Combine per-rule flags into one error code per row using the precedence.

    Args:
        flags: Boolean columns named by error code (``VL-004``, ``VL-006``, ``VL-001``,
            ``VL-002``, ``VL-003``); a missing column counts as never flagged.

    Returns:
        An object Series with the index of ``flags``: the highest-precedence flagged code
        (``VL-004``, ``VL-006``, ``VL-001``, ``VL-002``, ``VL-003``) or null when no rule flagged.
    """
    codes: pd.Series[str] = pd.Series(None, index=flags.index, dtype=object)
    for code in reversed(CODE_PRECEDENCE):  # lowest precedence first, so the highest wins
        if code in flags.columns:
            codes = codes.mask(flags[code].astype(bool), code)
    return codes


# --- report checks --------------------------------------------------------------------------


def _quoted(names: list[str]) -> str:
    """Join column names in single quotes.

    Args:
        names: Column names.

    Returns:
        The names as ``'a', 'b'``.
    """
    return ", ".join(f"'{name}'" for name in names)


def _cannot_check(name: str, missing: list[str]) -> CheckResult:
    """Build the failing result of a check whose input columns are missing.

    Args:
        name: Check name.
        missing: The missing column names.

    Returns:
        A failed `CheckResult` naming the missing columns.
    """
    return CheckResult(name, False, f"cannot check, missing column(s) {_quoted(missing)}")


def _list_result(name: str, offenders: list[str], ok_detail: str, bad_prefix: str) -> CheckResult:
    """Build a result that passes when ``offenders`` is empty, else names them.

    Args:
        name: Check name.
        offenders: Offending column names.
        ok_detail: Detail when nothing is offending.
        bad_prefix: Start of the detail when columns are offending.

    Returns:
        The `CheckResult`.
    """
    if not offenders:
        return CheckResult(name, True, ok_detail)
    return CheckResult(name, False, f"{bad_prefix} {_quoted(offenders)}")


def check_required_columns(frames: pd.DataFrame) -> CheckResult:
    """Check that every ``FRAMES_COLUMNS`` column is present.

    Args:
        frames: The ``frames`` table.

    Returns:
        The ``required_columns`` result; the detail names the missing columns.
    """
    missing = [name for name in FRAMES_COLUMNS if name not in frames.columns]
    ok = f"all {len(FRAMES_COLUMNS)} columns present"
    return _list_result(CHECK_REQUIRED_COLUMNS, missing, ok, "missing columns")


def check_dtypes(frames: pd.DataFrame) -> CheckResult:
    """Check that the present columns have the dtypes of ``FRAMES_DTYPES``.

    Args:
        frames: The ``frames`` table.

    Returns:
        The ``column_dtypes`` result; the detail names the offending columns.
    """
    wrong = [
        name
        for name, dtype in FRAMES_DTYPES.items()
        if name in frames.columns and str(frames[name].dtype) != dtype
    ]
    return _list_result(CHECK_DTYPES, wrong, "dtypes match", "unexpected dtype in")


def _has_null(frames: pd.DataFrame, column: str) -> bool:
    """Tell whether a key column holds a disallowed null.

    Args:
        frames: The ``frames`` table.
        column: A key column; the position columns may be null on VL-006 rows (empty cloud).

    Returns:
        ``True`` when the column is present and has a null that is not allowed.
    """
    if column not in frames.columns:
        return False
    values = frames[column]
    if column in POSITION_COLUMNS and "error_code" in frames.columns:
        values = values[frames["error_code"] != ErrorCode.SPARSE_CLOUD.value]
    return bool(values.isna().any())


def check_no_nulls(frames: pd.DataFrame) -> CheckResult:
    """Check that ``KEY_COLUMNS`` hold no null (positions may be null on VL-006 rows).

    Args:
        frames: The ``frames`` table.

    Returns:
        The ``no_nulls`` result; the detail names the columns with nulls. A row flagged VL-006
        may have a null ``x``, ``y``, ``z`` (an empty cloud has no centroid).
    """
    nulls = [name for name in KEY_COLUMNS if _has_null(frames, name)]
    return _list_result(CHECK_NO_NULLS, nulls, "no nulls in key columns", "nulls in")


def check_finite(frames: pd.DataFrame) -> CheckResult:
    """Check that no numeric column holds an infinite value (null is allowed).

    Args:
        frames: The ``frames`` table.

    Returns:
        The ``finite_values`` result; the detail names the offending columns.
    """
    numeric = frames.select_dtypes(include="number")
    infinite = [str(name) for name in numeric.columns if np.isinf(numeric[name]).any()]
    return _list_result(CHECK_FINITE, infinite, "all values finite", "infinite values in")


def check_ts_order(frames: pd.DataFrame) -> CheckResult:
    """Check that ``ts`` does not decrease per drive in ``frame_idx`` order.

    Rows flagged ``VL-004`` are exempt: they are reported through ``error_code_counts``.

    Args:
        frames: The ``frames`` table.

    Returns:
        The ``ts_non_decreasing`` result.
    """
    needed = ["drive_id", "frame_idx", "ts"]
    missing = [name for name in needed if name not in frames.columns]
    if missing:
        return _cannot_check(CHECK_TS_ORDER, missing)
    kept = frames
    if "error_code" in frames.columns:
        kept = frames[frames["error_code"] != ErrorCode.DUPLICATE_FRAME.value]
    ordered = kept.sort_values(["drive_id", "frame_idx"], kind="stable")
    backwards = int((ordered.groupby("drive_id")["ts"].diff() < 0).sum())
    return CheckResult(CHECK_TS_ORDER, backwards == 0, f"{backwards} backwards step(s)")


def check_error_code_counts(codes: pd.Series[str]) -> CheckResult:
    """Report how many frames carry each error code (always passes).

    Args:
        codes: The ``error_code`` values (null where a frame is clean).

    Returns:
        The ``error_code_counts`` result; the detail lists ``code=count`` pairs.
    """
    counts = codes.dropna().value_counts().sort_index()
    detail = ", ".join(f"{code}={count}" for code, count in counts.items()) or "no flagged frames"
    return CheckResult(CHECK_CODE_COUNTS, True, detail)


def check_flag_rate(codes: pd.Series[str], max_flag_rate: float) -> CheckResult:
    """Check that the share of flagged rows is at most ``max_flag_rate`` (inclusive).

    Args:
        codes: The ``error_code`` values (null where a frame is clean); may be empty.
        max_flag_rate: Largest allowed share of flagged rows.

    Returns:
        The ``flag_rate`` result; an empty input has a rate of 0.
    """
    rate = float(codes.notna().mean()) if len(codes) else 0.0
    detail = f"{rate:.2%} flagged (limit {max_flag_rate:.2%})"
    return CheckResult(CHECK_FLAG_RATE, rate <= max_flag_rate, detail)


def _codes_of(frames: pd.DataFrame) -> pd.Series[str]:
    """Return the ``error_code`` column, or an all-null column when it is missing.

    Args:
        frames: The ``frames`` table.

    Returns:
        The error codes, one per row.
    """
    if "error_code" in frames.columns:
        return frames["error_code"]
    return pd.Series(None, index=frames.index, dtype=object)


def validate_frames(
    frames: pd.DataFrame, cfg: Config = DEFAULT, max_flag_rate: float | None = None
) -> ValidationReport:
    """Check a derived ``frames`` table (several drives allowed).

    Runs one check per concern, in this order and under these names:

    - ``required_columns``: all ``FRAMES_COLUMNS`` present; the detail names the missing ones,
      each in single quotes.
    - ``column_dtypes``: present columns match ``FRAMES_DTYPES``; the detail names the offending
      columns in single quotes.
    - ``no_nulls``: no null in ``KEY_COLUMNS``; the detail names the offending columns in single
      quotes. ``x``, ``y``, ``z`` may be null on rows flagged ``VL-006`` (empty cloud).
    - ``finite_values``: no infinite value in any numeric column (null kinematics are allowed).
    - ``ts_non_decreasing``: per drive, in ``frame_idx`` order; rows flagged ``VL-004`` are
      exempt, they are reported through ``error_code_counts``.
    - ``error_code_counts``: always passes; the detail lists the count per code.
    - ``flag_rate``: share of rows with a non-null ``error_code`` is ``<= max_flag_rate``.

    An empty table has a flag rate of 0 (no division by zero) and passes every check. A check
    whose input column is missing fails with a detail naming it (``required_columns`` fails too).

    Args:
        frames: The ``frames`` table.
        cfg: Pipeline thresholds; ``cfg.max_flag_rate`` is the default flag-rate limit.
        max_flag_rate: Explicit override of the largest allowed share of flagged frames
            (``None`` means ``cfg.max_flag_rate``).

    Returns:
        The report; a missing column makes ``required_columns`` fail without raising.
    """
    codes = _codes_of(frames)
    limit = cfg.max_flag_rate if max_flag_rate is None else max_flag_rate
    return ValidationReport(
        (
            check_required_columns(frames),
            check_dtypes(frames),
            check_no_nulls(frames),
            check_finite(frames),
            check_ts_order(frames),
            check_error_code_counts(codes),
            check_flag_rate(codes, limit),
        )
    )


# --- realism --------------------------------------------------------------------------------


def _check_known(name: str, key: str, known: dict[str, object], label: str) -> CheckResult:
    """Check that ``key`` is one of the known names.

    Args:
        name: Check name.
        key: The profile or vehicle kind to look up.
        known: The registry whose keys are valid.
        label: What ``key`` is, for the detail (``profile`` or ``vehicle kind``).

    Returns:
        The `CheckResult`.
    """
    if key in known:
        return CheckResult(name, True, f"{label} '{key}'")
    return CheckResult(name, False, f"unknown {label} '{key}' (known: {', '.join(known)})")


def _usable_values(frames: pd.DataFrame, column: str) -> pd.Series[float] | str:
    """Return the non-null values of a column, or the reason it cannot be checked.

    Args:
        frames: The ``frames`` table of one drive.
        column: ``speed``, ``accel`` or ``yaw_rate``.

    Returns:
        The non-null values, or a message when the column is missing or has no value.
    """
    if column not in frames.columns:
        return f"cannot check, column '{column}' is missing"
    values = frames[column].dropna()
    if values.empty:
        return f"cannot check, column '{column}' has no non-null values"
    return values


def _max_speed_check(frames: pd.DataFrame, limits: ProfileLimits | None) -> CheckResult:
    """Check that the largest speed stays within the profile limit.

    Args:
        frames: The ``frames`` table of one drive.
        limits: The profile limits, or ``None`` for an unknown profile.

    Returns:
        The ``max_speed`` result.
    """
    if limits is None:
        return CheckResult(CHECK_MAX_SPEED, False, "unknown profile, no speed limit")
    speed = _usable_values(frames, "speed")
    if isinstance(speed, str):
        return CheckResult(CHECK_MAX_SPEED, False, speed)
    top = float(speed.max())
    detail = f"max speed {top:.2f} m/s (limit {limits.max_speed_ms} m/s)"
    return CheckResult(CHECK_MAX_SPEED, top <= limits.max_speed_ms, detail)


def _p99_check(
    name: str, frames: pd.DataFrame, column: str, limit: float | None, cfg: Config
) -> CheckResult:
    """Check that a percentile of ``abs(column)`` stays within ``limit``.

    Args:
        name: Check name.
        frames: The ``frames`` table of one drive.
        column: ``accel`` or ``yaw_rate``.
        limit: The limit, or ``None`` for an unknown profile.
        cfg: Pipeline thresholds; ``cfg.realism_percentile`` is the percentile compared.

    Returns:
        The `CheckResult`.
    """
    if limit is None:
        return CheckResult(name, False, "unknown profile, no limit")
    values = _usable_values(frames, column)
    if isinstance(values, str):
        return CheckResult(name, False, values)
    level = float(values.abs().quantile(cfg.realism_percentile))
    detail = f"p{cfg.realism_percentile * 100:g} {level:.3f} (limit {limit})"
    return CheckResult(name, level <= limit, detail)


def validate_realism(
    frames: pd.DataFrame, profile: str, vehicle_kind: str, cfg: Config = DEFAULT
) -> ValidationReport:
    """Check that the derived frames of one drive stay within the profile's physical limits.

    Runs, in this order, ``known_profile`` (``profile`` is a key of ``PROFILE_LIMITS``),
    ``known_vehicle_kind`` (``vehicle_kind`` is a key of ``VEHICLES``), ``max_speed``
    (``max(speed) <= PROFILE_LIMITS[profile].max_speed_ms``), ``accel_p99`` (99th percentile of
    ``abs(accel)`` over non-null values ``<= max_abs_accel_ms2``), ``yaw_rate_p99`` (same for
    ``abs(yaw_rate)`` against ``max_abs_yaw_rate_rad_s``) and ``flag_rate`` (share of flagged
    frames ``<= 0.05``). For an unknown profile ``max_speed``, ``accel_p99`` and
    ``yaw_rate_p99`` fail with a detail containing "unknown profile" instead of raising.

    Args:
        frames: The ``frames`` table of one drive.
        profile: A key of ``PROFILE_LIMITS``.
        vehicle_kind: A key of ``VEHICLES`` (only the ``known_vehicle_kind`` check uses it).
        cfg: Pipeline thresholds: ``cfg.realism_percentile`` is the percentile of the accel and
            yaw-rate checks (99th by default) and ``cfg.max_flag_rate`` the flag-rate limit.

    Returns:
        The report with the six checks above. A missing or all-null ``speed``, ``accel`` or
        ``yaw_rate`` column fails the check that needs it, naming the column, instead of raising.
    """
    limits = PROFILE_LIMITS.get(profile)
    accel_limit = limits.max_abs_accel_ms2 if limits else None
    yaw_limit = limits.max_abs_yaw_rate_rad_s if limits else None
    return ValidationReport(
        (
            _check_known(CHECK_KNOWN_PROFILE, profile, dict(PROFILE_LIMITS), "profile"),
            _check_known(CHECK_KNOWN_VEHICLE_KIND, vehicle_kind, dict(VEHICLES), "vehicle kind"),
            _max_speed_check(frames, limits),
            _p99_check(CHECK_ACCEL_P99, frames, "accel", accel_limit, cfg),
            _p99_check(CHECK_YAW_RATE_P99, frames, "yaw_rate", yaw_limit, cfg),
            check_flag_rate(_codes_of(frames), cfg.max_flag_rate),
        )
    )
