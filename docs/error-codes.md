# Error codes

Every frame that fails a quality rule gets exactly one error code in `frames.error_code`; clean frames
have `NULL`. The rules live in one module, [`src/vl/validation.py`](../src/vl/validation.py), used by
both the Spark transform (through `derive_frames`) and the standalone validator, so the two always
agree. The codes and their descriptions are the `ErrorCode` enum in
[`src/vl/errors.py`](../src/vl/errors.py). Every threshold is a field of `vl.config.Config`, overridable
with `VL_<FIELD>` (see the README).

## Summary

| Code | Name | Meaning | Tuned by |
|---|---|---|---|
| VL-001 | `DROPPED_FRAME` | Gap in `frame_idx` | nothing (structural) |
| VL-002 | `POSITION_JUMP` | Displacement between frames beyond the limit | `max_jump_m` (3.0 m) |
| VL-003 | `IMPOSSIBLE_ACCEL` | Derived acceleration beyond the physical limit | `max_accel_ms2` (15.0 m/s^2) |
| VL-004 | `DUPLICATE_FRAME` | Repeated `frame_idx`, or `ts` that does not increase | nothing (structural) |
| VL-005 | `MALFORMED_MESSAGE` | Message violates the message contract | nothing (contract) |
| VL-006 | `SPARSE_CLOUD` | Fewer points than the minimum, including an empty cloud | `min_cloud_points` (50) |

## Precedence

A frame can break several rules at once; it gets the highest-precedence code:

**VL-004 > VL-006 > VL-001 > VL-002 > VL-003**

VL-005 sits outside this order: a malformed message never becomes a frame. It's assigned at ingest and
goes to the `rejects` table.

## Detection rules

### VL-001 — Dropped frame

- **Rule:** `frame_idx` is more than 1 above the previous row's (`flag_frame_gaps`). One flag per gap,
  on the first frame after it, however many frames were lost. The first row is never flagged.
  Timestamps play no part in drop detection.
- **Assigned by:** the transform (`derive_frames`) and the validator.
- **The frame:** kept, with its kinematics computed from the last trusted frame using the real time
  difference, so its speed is correct. Not excluded from events by itself, but an event run stops at
  any flagged frame.
- **Example:** frames 100, 101, 105 → frame 105 gets VL-001. A malformed message also leaves a gap, so
  the frame after it gets VL-001 too (the generator labels it `gap_after_malformed`).

### VL-002 — Position jump

- **Rule:** `hypot(dx, dy)` of the centroid, against the previous row that has a position, exceeds
  `max_jump_m` (`flag_position_jump`). A gap in `frame_idx` doesn't matter. The first row and rows with
  a null position are never flagged.
- **Assigned by:** the transform and the validator.
- **The frame:** kinematics (`vx, vy, speed, accel, yaw_rate`) are nulled, and the neighbours are left
  uncontaminated. The jump is a persistent offset, so later frames use the jumped frame as their
  baseline and stay correct.
- **Example:** the simulator shifts a frame and all later ones by 5 m along x → that frame is VL-002,
  the rest are clean.

### VL-003 — Impossible acceleration

- **Rule:** `abs(accel) > max_accel_ms2` (`flag_impossible_accel`), evaluated last, on the acceleration
  derived from the smoothed speed. Null acceleration is never flagged.
- **Assigned by:** the transform and the validator.
- **The frame:** keeps its kinematics (it doesn't make the row suspect) but is flagged, so it's
  excluded from events.
- **Note:** the simulator's drives never trigger it (it didn't occur in the review's fleet runs); the
  rule is a guard for real data.

### VL-004 — Duplicate or out-of-order frame

- **Rule:** the `frame_idx` occurs more than once (every occurrence is flagged), or the `ts` is less
  than or equal to the previous row's in `frame_idx` order (`flag_duplicate_or_backwards_ts`). Rows are
  sorted by `frame_idx` first, so a frame that merely arrives late is put back in order and isn't
  flagged; a `ts` that runs backwards is.
- **Assigned by:** the transform and the validator.
- **The frame:** one row per `frame_idx` is kept in `frames` and carries the code — the duplicates are
  collapsed, so `frames` can have fewer rows than `raw_frames`. Its kinematics are null and it's never
  used as the baseline of a later frame. In the Spark transform the kept duplicate is deterministic:
  the one with the lowest `ts`.
- **Example:** two messages with `frame_idx` 500 → one row with VL-004 in `frames`. A micro-batch Spark
  replays after a crash doesn't create duplicates: the ingest's batch writes are idempotent (see the
  architecture doc's reliability section).

### VL-005 — Malformed message

- **Rule:** the Kafka message violates the strict contract (see
  [architecture.md](architecture.md#message-contract)): not exactly one JSON object, `drive_id` not a
  string, `frame_idx` not an integer, `ts` not a finite number, `points` not an array of integer
  triples, a metadata field that isn't a string, a blank line, trailing text, single quotes. For a v2
  message (`schema_version` 2): `centroid_m` not three finite numbers (or present on an empty cloud),
  `n_points` not an integer from 0 to 2,147,483,647, `cloud_ref` missing or not a string. And any
  `schema_version` other than absent, null, `1` or `2`.
- **Assigned by:** the Spark ingest (`vl.etl.ingest`); the validator CLI applies the same contract to
  JSONL lines — blank lines, `NaN` tokens, int64 overflow, absurdly deep nesting — and counts them in
  `error_code_counts` and `flag_rate`. The Parquet route (`raw_frames`) has no VL-005: rejected messages
  live in `rejects`, which it doesn't read.
- **The message:** goes to `rejects` (`payload` verbatim, `error_code`, `ingested_at`), not to
  `raw_frames` or `frames`, because it has no usable `drive_id`.
- **Example:** `{"drive_id": "drive-002", ..., "points": "garbage"}`.

### VL-006 — Empty or sparse point cloud

- **Rule:** `n_points < min_cloud_points` (`flag_sparse_cloud`). An empty cloud (`points: []`) is a
  valid message with `n_points = 0` and a null centroid; at the default threshold it's VL-006.
- **Assigned by:** the transform and the validator.
- **The frame:** kept; kinematics are null, and it's never a baseline for a later frame. Its `x, y, z`
  may be null — the only place the key columns are allowed to be.
- **Example:** a frame with 10 points instead of 200 (the simulator's sparse fault).

## Events

Events are detected per drive on clean frames only (`error_code IS NULL`). A run of frames meeting the
condition must last at least `min_event_frames` (4) consecutive frames and stops at any flagged frame.
Event types are the `EventType` enum.

| Event | Condition (default threshold) | Config key | `params` JSON |
|---|---|---|---|
| `HARD_BRAKE` | `accel <= -3.5` m/s^2 | `hard_brake_ms2` | `{"min_accel": ...}` |
| `HARD_ACCEL` | `accel >= 3.0` m/s^2 | `hard_accel_ms2` | `{"max_accel": ...}` |
| `SHARP_TURN` | `abs(yaw_rate) >= 0.35` rad/s | `sharp_turn_rad_s` | `{"max_abs_yaw_rate": ...}` |
| `STOP` | `speed < 0.2` m/s for at least 2.0 s | `stop_speed_ms`, `stop_min_s` | `{"duration_s": ...}` |

The STOP duration is `n_frames * frame_period_s`, so at the defaults 31 frames are not a stop and 32
are. The yaw rate is 0 below `yaw_min_speed_ms` (0.5 m/s). `start_frame` and `end_frame` are both
inclusive.

Events are built on trusted frames by design. A flagged frame inside a maneuver ends the run: a VL-003
frame is a physically impossible acceleration, and after a VL-001 gap the motion inside the gap is
unknown. So one real hard brake that touches a flagged frame is reported as two shorter events, or as
none when each part falls below `min_event_frames`.

## Validator CLI

`python -m vl.validate --source ... [--labels ... [--realism]]` prints one line per check,
`<PASS|FAIL> <name>: <detail>`, and exits with:

| Exit code | Meaning |
|---|---|
| 0 | every check passed |
| 1 | at least one check failed |
| 2 | unusable input: a missing, unreadable or empty-directory source, a bad `labels.json`, `--realism` without `--labels`, a source drive missing from the labels' `drives`, or an invalid `VL_*` value (one `error:` line on stderr). Argparse usage errors also exit 2. |

Checks: `source_not_empty`, `duplicates_collapsed` (always passes — raw rows dropped because their
`frame_idx` repeated), `frames_lost` (always passes — frames missing inside the `frame_idx` sequences,
and the number of gaps, since VL-001 flags one row per gap however many frames it lost),
`required_columns`, `column_dtypes`, `no_nulls`, `finite_values`, `ts_non_decreasing`,
`error_code_counts` (always passes), `flag_rate` (limit `max_flag_rate`). With `--labels`:
`labeled_faults_detected` (every labeled fault got its expected code) and `false_positive_rate` (limit
`max_false_positive_rate`). With `--realism`, per drive `realism[<drive_id>].<check>` for
`known_profile`, `known_vehicle_kind`, `max_speed`, `accel_p99`, `yaw_rate_p99` and `flag_rate`.

`flag_rate` is the flagged share of the received rows. On a JSONL source the malformed lines count as
VL-005 in both numerator and denominator. On a Parquet `raw_frames` source the rejects aren't read, so
VL-005 is absent. On either source, frames that never arrived (see `frames_lost`) aren't in the rate.
A stream with many gaps can therefore show a low `flag_rate` — read it together with `frames_lost`.

The other CLIs (`transform`, `ingest`, `producer`) exit 0 on success, 1 when Kafka can't be reached or
the topic is missing (ingest, producer), and 2 for unusable arguments or paths (argparse errors; for
`transform`, a warehouse path that is a file, or an invalid `VL_*` value).
