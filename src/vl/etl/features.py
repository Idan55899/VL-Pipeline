"""Ego-motion features, event detection and drive summaries for one drive at a time.

Role in the pipeline: the pure transform behind the ``frames``, ``events`` and ``drives``
tables. ``derive_frames`` composes the error-code rules of `vl.validation` with the
kinematics; no rule logic lives here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd

from vl.config import DEFAULT, Config
from vl.errors import ErrorCode, EventType
from vl.schemas import (
    DRIVES_COLUMNS,
    DRIVES_DTYPES,
    EVENTS_COLUMNS,
    EVENTS_DTYPES,
    FRAMES_COLUMNS,
    FRAMES_DTYPES,
    RAW_FRAMES_COLUMNS,
)
from vl.validation import (
    assign_error_codes,
    flag_duplicate_or_backwards_ts,
    flag_frame_gaps,
    flag_impossible_accel,
    flag_position_jump,
    flag_sparse_cloud,
)

# Codes whose frames are excluded from kinematics, and those whose timestamp or centroid is so
# untrusted that the frame cannot serve as the baseline of a later velocity.
SUSPECT_CODES = (
    ErrorCode.POSITION_JUMP.value,
    ErrorCode.DUPLICATE_FRAME.value,
    ErrorCode.SPARSE_CLOUD.value,
)
UNTRUSTED_BASELINE_CODES = (ErrorCode.DUPLICATE_FRAME.value, ErrorCode.SPARSE_CLOUD.value)
METADATA_COLUMNS = ["vehicle_id", "vehicle_kind", "drive_start_utc"]
KINEMATIC_COLUMNS = ["vx", "vy", "speed", "accel", "yaw_rate"]


# --- derive_frames --------------------------------------------------------------------------


def _empty_frames() -> pd.DataFrame:
    """Build an empty ``frames`` table.

    Returns:
        A DataFrame with ``FRAMES_COLUMNS`` and ``FRAMES_DTYPES`` and no rows.
    """
    return pd.DataFrame(columns=FRAMES_COLUMNS).astype(FRAMES_DTYPES)


def _check_single_drive(raw: pd.DataFrame) -> None:
    """Reject input that holds more than one drive.

    Args:
        raw: Raw frames.

    Raises:
        ValueError: If ``drive_id`` has more than one distinct non-null value; the message
            names them.
    """
    if "drive_id" not in raw.columns:
        return
    drives = sorted(str(drive) for drive in raw["drive_id"].dropna().unique())
    if len(drives) > 1:
        raise ValueError(f"derive_frames takes one drive at a time, got {len(drives)}: {drives}")


def _coerce(table: pd.DataFrame) -> pd.DataFrame:
    """Coerce the numeric columns of a raw table to the documented dtypes.

    Object-dtype and nullable (``Int64``, ``Float64``) inputs become ``int64`` for
    ``frame_idx`` and ``n_points`` and ``float64`` for ``ts, x, y, z``; a null centroid
    (empty cloud) becomes NaN.

    Args:
        table: Raw frames with all ``RAW_FRAMES_COLUMNS``.

    Returns:
        A copy with the coerced columns.

    Raises:
        ValueError: If a value cannot be converted (for example a null ``frame_idx``).
    """
    out = table.copy()
    for column in ("ts", "x", "y", "z"):
        out[column] = pd.to_numeric(out[column]).astype("float64")
    for column in ("frame_idx", "n_points"):
        out[column] = pd.to_numeric(out[column]).astype("int64")
    return out


def _prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series[bool]]:
    """Sort by ``frame_idx``, flag VL-004 and keep the first of duplicate ``frame_idx``.

    The sort is stable, so among duplicates the first row in the given order is the one kept
    (see `derive_frames` for which row that is on each route). The VL-004 flag is computed
    before the duplicates are dropped (every occurrence is flagged) and the kept row carries it.

    Args:
        raw: Raw frames of one drive; missing optional columns are added as nulls.

    Returns:
        The deduplicated table (positional index) and its VL-004 flags, aligned to it.

    Raises:
        ValueError: If the input holds more than one drive or a value cannot be coerced.
    """
    _check_single_drive(raw)
    table = _coerce(raw.reindex(columns=RAW_FRAMES_COLUMNS))
    for column in METADATA_COLUMNS:  # absent metadata becomes None, not float NaN
        table[column] = table[column].astype(object).where(table[column].notna(), None)
    table = table.sort_values("frame_idx", kind="stable").reset_index(drop=True)
    backwards = flag_duplicate_or_backwards_ts(table["frame_idx"], table["ts"])
    keep = ~table["frame_idx"].duplicated(keep="first")
    return table[keep].reset_index(drop=True), backwards[keep].reset_index(drop=True)


def _rule_flags(table: pd.DataFrame, duplicate: pd.Series[bool], cfg: Config) -> pd.DataFrame:
    """Apply the position-independent rules of `vl.validation` to a prepared table.

    VL-003 is not here: it needs the acceleration, which needs the kinematics.

    Args:
        table: The prepared table (sorted, deduplicated).
        duplicate: The VL-004 flags of ``table``.
        cfg: Pipeline thresholds.

    Returns:
        A boolean DataFrame with the columns ``VL-004``, ``VL-006``, ``VL-001``, ``VL-002``.
    """
    return pd.DataFrame(
        {
            ErrorCode.DUPLICATE_FRAME.value: duplicate,
            ErrorCode.SPARSE_CLOUD.value: flag_sparse_cloud(table["n_points"], cfg),
            ErrorCode.DROPPED_FRAME.value: flag_frame_gaps(table["frame_idx"]),
            ErrorCode.POSITION_JUMP.value: flag_position_jump(table["x"], table["y"], cfg),
        }
    )


def _raw_velocity(table: pd.DataFrame, codes: pd.Series[str]) -> pd.DataFrame:
    """Estimate the unsmoothed velocity of every row from its baseline row.

    The baseline of row ``i`` is the nearest previous row not flagged VL-004 or VL-006 (their
    timestamp or centroid is untrusted). A VL-002 row is a valid baseline: a jump is a
    persistent offset, so the row after it is computed against the shifted position. The
    velocity is ``(pos_i - pos_baseline) / (ts_i - ts_baseline)`` with the real timestamps,
    so a row after a dropped frame still gets the right speed. Rows with no baseline or a
    non-positive time step are null.

    Args:
        table: The prepared table.
        codes: Error codes before VL-003 (nulls where clean).

    Returns:
        A DataFrame aligned to ``table`` with ``vx``, ``vy``, ``dt`` (the time step the
        velocity covers, which weights the smoothing) and ``mid`` (the middle of that step,
        the time the velocity really belongs to).
    """
    trusted = ~codes.isin(UNTRUSTED_BASELINE_CODES)
    baseline = table[["ts", "x", "y"]].where(trusted).ffill().shift(1)
    dt = (table["ts"] - baseline["ts"]).where(lambda step: step > 0)
    return pd.DataFrame(
        {
            "vx": (table["x"] - baseline["x"]) / dt,
            "vy": (table["y"] - baseline["y"]) / dt,
            "dt": dt,
            "mid": (table["ts"] + baseline["ts"]) / 2,
        }
    )


def _hypot(a: pd.Series[float], b: pd.Series[float]) -> pd.Series[float]:
    """Compute ``sqrt(a^2 + b^2)`` elementwise (null where either is null).

    Args:
        a: First leg.
        b: Second leg, aligned to ``a``.

    Returns:
        The hypotenuse as a Series with the index of ``a``.
    """
    return pd.Series(np.hypot(a, b), index=a.index)


def _smooth(values: pd.Series[float], weights: pd.Series[float], window: int) -> pd.Series[float]:
    """Smooth with a centered, time-weighted moving average over ``window`` rows.

    Rows where ``values`` or ``weights`` is null are ignored. Each value is weighted by the
    time step it covers, so the average is the displacement across the window divided by its
    duration: a velocity that spans a dropped-frame gap counts for the gap's whole length
    instead of as one frame (an unweighted mean would bias speed and acceleration there). The
    window is centered so the smoothed signal stays aligned in time with the raw one (a
    trailing window would delay every event by half a window).

    Args:
        values: The raw values, null where undefined.
        weights: The time step each value covers, aligned to ``values``.
        window: Window size in rows (``cfg.smooth_frames``).

    Returns:
        The weighted mean of the usable values inside the window (null when there are none).
    """
    usable = values.notna() & weights.notna()
    total = (values * weights).where(usable).rolling(window, center=True, min_periods=1).sum()
    duration = weights.where(usable).rolling(window, center=True, min_periods=1).sum()
    return total / duration


def _steps(values: pd.Series[float], ts: pd.Series[float]) -> pd.Series[float]:
    """Time step between each non-null value and the previous non-null one.

    Args:
        values: Values, null where undefined.
        ts: Timestamps aligned to ``values``.

    Returns:
        The step in seconds per row; null where there is no previous value or the step is not
        positive.
    """
    known = values.notna()
    return ts[known].diff().where(lambda dt: dt > 0).reindex(values.index)


def _derivative(values: pd.Series[float], ts: pd.Series[float]) -> pd.Series[float]:
    """Differentiate against ``ts`` between consecutive non-null values.

    Using the previous non-null value (not the previous row) keeps the derivative correct next
    to a null row, with the real time step between the two samples.

    Args:
        values: Values, null where undefined.
        ts: Timestamps aligned to ``values``.

    Returns:
        ``d values / d ts`` per row; null where there is no previous value or the step is not
        positive.
    """
    change = values[values.notna()].diff().reindex(values.index)
    return change / _steps(values, ts)


def _yaw_rate(
    vx: pd.Series[float], vy: pd.Series[float], ts: pd.Series[float], cfg: Config
) -> pd.Series[float]:
    """Derive the yaw rate from the unwrapped heading of the smoothed velocity.

    The heading is unwrapped over the rows that have a velocity, differentiated, and smoothed
    over ``window`` rows like the velocity: differentiating a heading that is itself a short
    moving average still leaves enough noise at low speed to fake a turn, and averaging the
    derivative over the same window cuts it by about four with a delay-free, centered window.
    The rate is 0 where the speed is below ``cfg.yaw_min_speed_ms``, and also on the first moving
    row after a slow stretch or at the start (no trustworthy previous heading, so no spurious
    spike when a stopped vehicle starts moving). Rows without a velocity stay null.

    Args:
        vx: Smoothed x velocity, null where undefined.
        vy: Smoothed y velocity, null where undefined.
        ts: Timestamps aligned to the velocities.
        cfg: Pipeline thresholds (``smooth_frames`` window, ``yaw_min_speed_ms`` floor).

    Returns:
        The yaw rate in rad/s per row.
    """
    speed = _hypot(vx, vy)
    heading = pd.Series(np.arctan2(vy, vx), index=vx.index)
    known = heading.notna()
    unwrapped = pd.Series(np.nan, index=vx.index)
    unwrapped[known] = np.unwrap(heading[known].to_numpy())
    moving = (speed >= cfg.yaw_min_speed_ms)[known]
    both_moving = (moving & moving.shift(1, fill_value=False)).reindex(vx.index, fill_value=False)
    rate = _smooth(_derivative(unwrapped, ts), _steps(unwrapped, ts), cfg.smooth_frames)
    return rate.where(both_moving, 0.0).where(known)


def _kinematics(table: pd.DataFrame, codes: pd.Series[str], cfg: Config) -> pd.DataFrame:
    """Compute ``vx, vy, speed, accel, yaw_rate`` for every row.

    Raw velocities come from `_raw_velocity`, are smoothed over ``cfg.smooth_frames`` rows
    (time-weighted, ignoring nulls), and are then nulled again on suspect rows (VL-002, VL-004,
    VL-006) and on rows that had no raw velocity (the first row), so smoothing never invents a
    value for them nor lets a suspect row contaminate its neighbours. Each smoothed value is
    dated at the weighted mean of the times its raw velocities belong to (``tau``), and
    ``accel`` and ``yaw_rate`` differentiate against ``tau``, so a dropped-frame gap neither
    biases the speed nor fakes an acceleration spike.

    Args:
        table: The prepared table.
        codes: Error codes before VL-003 (nulls where clean).
        cfg: Pipeline thresholds.

    Returns:
        A DataFrame with the ``KINEMATIC_COLUMNS``, aligned to ``table``.
    """
    raw = _raw_velocity(table, codes)
    valid = raw["vx"].notna() & raw["vy"].notna() & ~codes.isin(SUSPECT_CODES)
    weights = raw["dt"].where(valid)

    def smooth(column: str) -> pd.Series[float]:
        """Smooth one raw column over the window, null on invalid rows."""
        return _smooth(raw[column].where(valid), weights, cfg.smooth_frames).where(valid)

    vx, vy, tau = smooth("vx"), smooth("vy"), smooth("mid")
    speed = _hypot(vx, vy)
    accel = _derivative(speed, tau)
    yaw = _yaw_rate(vx, vy, tau, cfg)
    return pd.DataFrame({"vx": vx, "vy": vy, "speed": speed, "accel": accel, "yaw_rate": yaw})


def derive_frames(raw: pd.DataFrame, cfg: Config = DEFAULT) -> pd.DataFrame:
    """Turn the raw frames of one drive into the ``frames`` table.

    Sorts by ``frame_idx``; keeps one row of each duplicated ``frame_idx`` and flags it
    ``VL-004`` (a ``ts`` that does not increase is also ``VL-004``). The sort is stable, so
    the kept duplicate is the first emitted one on the JSONL route; the transform and the
    Parquet validator route sort the rows by ``RAW_ORDER_COLUMNS`` first, so there the kept
    duplicate is the one with the lowest ``ts`` (then the lowest following column). Error codes
    come from the rule functions in `vl.validation` with the precedence VL-004, VL-006, VL-001,
    VL-002, VL-003.

    Kinematics (baseline rows, smoothing, derivatives) are documented in `_raw_velocity`,
    `_kinematics` and `_yaw_rate`. Suspect rows (``VL-002``, ``VL-004``, ``VL-006``) get null
    ``vx, vy, speed, accel, yaw_rate``; the row after a suspect row is computed from its baseline
    and so stays correct (no false ``VL-003``). The first row has null speed and acceleration. The
    columns ``vehicle_id``, ``vehicle_kind`` and ``drive_start_utc`` pass through (null when
    ``raw`` lacks them).

    Because suspect status depends on the final code, a row flagged both VL-001 and VL-002 gets
    VL-001 (precedence) and is not suspect: after a long gap the displacement is real motion.
    VL-003 is evaluated last, on the acceleration, and does not make a row suspect.

    An empty cloud (``n_points == 0``, null ``x, y, z``) is a normal VL-006 row: it is kept,
    keeps its null position and is a suspect row. Object-dtype and nullable-integer inputs are
    coerced to the documented dtypes.

    Known limitation: a timestamp spiking forward is not flagged on its own row (the VL-004
    rule flags the row after it), so the spike row and its neighbours get wrong velocities.

    Args:
        raw: Raw frames of one drive (``RAW_FRAMES_COLUMNS``; the metadata columns are
            optional). May be empty and may have no columns at all.
        cfg: Pipeline thresholds.

    Returns:
        A DataFrame with exactly ``FRAMES_COLUMNS`` (dtypes ``FRAMES_DTYPES``), one row per
        distinct ``frame_idx``, sorted by ``frame_idx``; empty input gives an empty frame.

    Raises:
        ValueError: If ``raw`` holds more than one drive (the message names the ``drive_id``
            values) or a numeric value cannot be coerced.
    """
    if raw.empty:
        return _empty_frames()
    table, duplicate = _prepare(raw)
    flags = _rule_flags(table, duplicate, cfg)
    kinematics = _kinematics(table, assign_error_codes(flags), cfg)
    flags[ErrorCode.IMPOSSIBLE_ACCEL.value] = flag_impossible_accel(kinematics["accel"], cfg)
    frames = table.join(kinematics)
    frames["error_code"] = assign_error_codes(flags)
    return frames[FRAMES_COLUMNS].astype(FRAMES_DTYPES)


# --- detect_events --------------------------------------------------------------------------


class _EventSpec(NamedTuple):
    """How to detect one event type.

    Attributes:
        event_type: The event type (written to the table as its string value).
        holds: Per-row condition on the frames table.
        param: Name of the ``params`` key.
        measure: Value of the ``params`` key for the run's rows.
        is_stop: Whether the run also needs ``cfg.stop_min_s`` of duration.
    """

    event_type: EventType
    holds: Callable[[pd.DataFrame, Config], pd.Series[bool]]
    param: str
    measure: Callable[[pd.DataFrame, Config], float]
    is_stop: bool = False


_EVENT_SPECS = (
    _EventSpec(
        EventType.HARD_BRAKE,
        lambda f, c: f["accel"] <= c.hard_brake_ms2,
        "min_accel",
        lambda run, c: float(run["accel"].min()),
    ),
    _EventSpec(
        EventType.HARD_ACCEL,
        lambda f, c: f["accel"] >= c.hard_accel_ms2,
        "max_accel",
        lambda run, c: float(run["accel"].max()),
    ),
    _EventSpec(
        EventType.SHARP_TURN,
        lambda f, c: f["yaw_rate"].abs() >= c.sharp_turn_rad_s,
        "max_abs_yaw_rate",
        lambda run, c: float(run["yaw_rate"].abs().max()),
    ),
    _EventSpec(
        EventType.STOP,
        lambda f, c: f["speed"] < c.stop_speed_ms,
        "duration_s",
        lambda run, c: len(run) * c.frame_period_s,
        is_stop=True,
    ),
)


def _runs(mask: pd.Series[bool]) -> list[tuple[int, int]]:
    """Find the runs of consecutive ``True`` values.

    Args:
        mask: A boolean Series (nulls count as ``False``).

    Returns:
        ``(first, last)`` row positions of each run, both inclusive, in order.
    """
    padded = np.concatenate([[0], mask.fillna(False).to_numpy(dtype=int), [0]])
    edges = np.diff(padded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _long_enough(first: int, last: int, spec: _EventSpec, cfg: Config) -> bool:
    """Tell whether a run is long enough to be an event.

    Args:
        first: First row position of the run.
        last: Last row position of the run (inclusive).
        spec: The event type.
        cfg: Pipeline thresholds.

    Returns:
        ``True`` when the run has ``cfg.min_event_frames`` rows and, for a stop, lasts
        ``cfg.stop_min_s`` (duration ``n_frames * cfg.frame_period_s``).
    """
    n_frames = last - first + 1
    if n_frames < cfg.min_event_frames:
        return False
    return not spec.is_stop or n_frames * cfg.frame_period_s >= cfg.stop_min_s


def _event_row(
    frames: pd.DataFrame, first: int, last: int, spec: _EventSpec, cfg: Config
) -> dict[str, object]:
    """Build one ``events`` row from a run.

    Args:
        frames: The frames table.
        first: First row position of the run.
        last: Last row position of the run (inclusive).
        spec: The event type.
        cfg: Pipeline thresholds.

    Returns:
        A mapping with the ``EVENTS_COLUMNS`` keys.
    """
    run = frames.iloc[first : last + 1]
    return {
        "drive_id": run["drive_id"].iloc[0],
        "event_type": spec.event_type.value,
        "start_frame": int(run["frame_idx"].iloc[0]),
        "end_frame": int(run["frame_idx"].iloc[-1]),
        "start_ts": float(run["ts"].iloc[0]),
        "end_ts": float(run["ts"].iloc[-1]),
        "params": json.dumps({spec.param: spec.measure(run, cfg)}),
    }


def _events_of(frames: pd.DataFrame, spec: _EventSpec, cfg: Config) -> list[dict[str, object]]:
    """Detect the events of one type.

    Args:
        frames: The frames table of one drive.
        spec: The event type.
        cfg: Pipeline thresholds.

    Returns:
        One row mapping per run that meets the condition on clean frames only.
    """
    mask = spec.holds(frames, cfg) & frames["error_code"].isna()
    runs = [run for run in _runs(mask) if _long_enough(*run, spec, cfg)]
    return [_event_row(frames, first, last, spec, cfg) for first, last in runs]


def detect_events(frames: pd.DataFrame, cfg: Config = DEFAULT) -> pd.DataFrame:
    """Detect HARD_BRAKE, HARD_ACCEL, SHARP_TURN and STOP events in a derived drive.

    A run needs at least ``cfg.min_event_frames`` consecutive frames meeting the condition
    (``accel <= cfg.hard_brake_ms2``, ``accel >= cfg.hard_accel_ms2``,
    ``abs(yaw_rate) >= cfg.sharp_turn_rad_s``, ``speed < cfg.stop_speed_ms``) and runs break
    at any frame with a non-null ``error_code``. ``STOP`` additionally needs a duration of at
    least ``cfg.stop_min_s``, where the duration is ``n_frames * cfg.frame_period_s`` (with the
    defaults 31 frames are not a stop and 32 are). ``start_frame``/``end_frame`` are the first
    and last frame of the run, both inclusive. ``params`` is a JSON string carrying the extreme
    measured value: HARD_BRAKE ``{"min_accel": ...}``, HARD_ACCEL ``{"max_accel": ...}``,
    SHARP_TURN ``{"max_abs_yaw_rate": ...}``, STOP ``{"duration_s": ...}``.

    Known limitation: there is no hysteresis, so a signal hovering around a threshold can split
    one maneuver into several shorter events (or drop it below ``cfg.min_event_frames``).

    The runs are found without a per-row loop (differences of the padded condition mask); only
    the assembly of one output row per run loops. A null kinematic value never meets a
    condition, and flagged frames are never part of a run, so a run cannot straddle a gap.

    Args:
        frames: The ``frames`` table of one drive.
        cfg: Pipeline thresholds.

    Returns:
        A DataFrame with exactly ``EVENTS_COLUMNS``, sorted by ``start_frame``.
    """
    rows = [row for spec in _EVENT_SPECS for row in _events_of(frames, spec, cfg)]
    events = pd.DataFrame(rows, columns=EVENTS_COLUMNS).astype(EVENTS_DTYPES)
    return events.sort_values("start_frame", kind="stable").reset_index(drop=True)


# --- summarize_drive ------------------------------------------------------------------------


def _distance(frames: pd.DataFrame) -> float:
    """Integrate the speed over the retained frames.

    Each row contributes ``speed * (ts - previous ts)``; a null speed (first row, suspect
    rows) is first filled by interpolation so a flagged frame does not lose its share of the
    distance. A single row has no time step and gives 0.

    Args:
        frames: The frames table of one drive.

    Returns:
        The distance in meters.
    """
    speed = frames["speed"].interpolate(limit_direction="both")
    step = frames["ts"].diff().clip(lower=0.0)
    return float((speed * step).fillna(0.0).sum())


def summarize_drive(frames: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Summarize one drive into a single ``drives`` row.

    ``duration_s`` is the elapsed time of the retained frames (last ``ts`` minus first ``ts``,
    not ``n_frames * frame_period_s``), ``start_ts`` the first ``ts``, ``distance_m`` about the
    integral of speed over the drive (see `_distance`), ``n_events`` the number of events,
    ``n_errors`` the number of frames with a non-null ``error_code``, ``n_frames`` the number of
    frame rows.

    Args:
        frames: The ``frames`` table of one drive (may be empty).
        events: The ``events`` table of the same drive.

    Returns:
        A one-row DataFrame with exactly ``DRIVES_COLUMNS``; an empty frame with
        ``DRIVES_COLUMNS`` when ``frames`` is empty.
    """
    if frames.empty:
        return pd.DataFrame(columns=DRIVES_COLUMNS).astype(DRIVES_DTYPES)
    first = frames.iloc[0]
    row = {name: first[name] for name in ["drive_id", *METADATA_COLUMNS]}
    row.update(
        start_ts=float(first["ts"]),
        duration_s=float(frames["ts"].iloc[-1] - first["ts"]),
        n_frames=len(frames),
        distance_m=_distance(frames),
        n_events=len(events),
        n_errors=int(frames["error_code"].notna().sum()),
    )
    return pd.DataFrame([row], columns=DRIVES_COLUMNS).astype(DRIVES_DTYPES)
