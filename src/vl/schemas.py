"""Table schemas of the pipeline: Spark DDL strings, column lists and pandas dtypes.

Role in the pipeline: the single source of truth for column names and order. The ETL
(``raw_frames``, ``frames``, ``events``, ``drives``, ``rejects``), the validator and the
dashboard all read the columns defined here; the column orders are binding.
"""

RAW_FRAMES_DDL = (
    "drive_id string, vehicle_id string, vehicle_kind string, drive_start_utc string, "
    "frame_idx long, ts double, n_points int, x double, y double, z double"
)
FRAMES_DDL = (
    "drive_id string, vehicle_id string, vehicle_kind string, drive_start_utc string, "
    "frame_idx long, ts double, x double, y double, z double, vx double, vy double, "
    "speed double, accel double, yaw_rate double, error_code string"
)
EVENTS_DDL = (
    "drive_id string, event_type string, start_frame long, end_frame long, "
    "start_ts double, end_ts double, params string"
)
DRIVES_DDL = (
    "drive_id string, vehicle_id string, vehicle_kind string, drive_start_utc string, "
    "start_ts double, duration_s double, n_frames long, distance_m double, "
    "n_events long, n_errors long"
)
REJECTS_DDL = "payload string, error_code string, ingested_at timestamp"

# Table name -> Spark DDL string, for code that handles the tables generically (the transform
# writers and the DuckDB views).
TABLE_DDLS: dict[str, str] = {
    "frames": FRAMES_DDL,
    "events": EVENTS_DDL,
    "drives": DRIVES_DDL,
    "rejects": REJECTS_DDL,
    "raw_frames": RAW_FRAMES_DDL,
}

# Sort key that makes the raw rows of a drive a total order. ``raw_frames`` has no
# emission-order column and its rows arrive in arbitrary order, so conflicting copies of one
# ``frame_idx`` are resolved the same way on every run (the first row after sorting is kept).
RAW_ORDER_COLUMNS = [
    "frame_idx",
    "ts",
    "n_points",
    "x",
    "y",
    "z",
    "vehicle_id",
    "vehicle_kind",
    "drive_start_utc",
]
RAW_FRAMES_COLUMNS = [
    "drive_id",
    "vehicle_id",
    "vehicle_kind",
    "drive_start_utc",
    "frame_idx",
    "ts",
    "n_points",
    "x",
    "y",
    "z",
]
# Message fields that must be JSON strings (``null`` or absent allowed except ``drive_id``; v2's
# ``cloud_ref`` is also required in a v2 message), the message version that carries a centroid and
# a cloud reference instead of the points, and the range of the JSON integers the message contract
# accepts (``long`` in the Spark ingest).
MESSAGE_STRING_FIELDS = (*RAW_FRAMES_COLUMNS[:4], "cloud_ref")
SCHEMA_V2 = 2
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
MAX_POINTS = 2**31 - 1  # largest v2 ``n_points``: the ``n_points`` column is a 32-bit ``int``
FRAMES_COLUMNS = [
    "drive_id",
    "vehicle_id",
    "vehicle_kind",
    "drive_start_utc",
    "frame_idx",
    "ts",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "speed",
    "accel",
    "yaw_rate",
    "error_code",
]
EVENTS_COLUMNS = [
    "drive_id",
    "event_type",
    "start_frame",
    "end_frame",
    "start_ts",
    "end_ts",
    "params",
]
DRIVES_COLUMNS = [
    "drive_id",
    "vehicle_id",
    "vehicle_kind",
    "drive_start_utc",
    "start_ts",
    "duration_s",
    "n_frames",
    "distance_m",
    "n_events",
    "n_errors",
]
REJECTS_COLUMNS = ["payload", "error_code", "ingested_at"]

# pandas dtype of every ``frames`` column in a DataFrame produced by ``derive_frames``:
# strings (including all-null metadata and ``error_code``) are ``object``.
FRAMES_DTYPES = {
    "drive_id": "object",
    "vehicle_id": "object",
    "vehicle_kind": "object",
    "drive_start_utc": "object",
    "frame_idx": "int64",
    "ts": "float64",
    "x": "float64",
    "y": "float64",
    "z": "float64",
    "vx": "float64",
    "vy": "float64",
    "speed": "float64",
    "accel": "float64",
    "yaw_rate": "float64",
    "error_code": "object",
}

# pandas dtypes of the ``events`` and ``drives`` tables (also when they are empty), matching
# EVENTS_DDL and DRIVES_DDL: strings are ``object``, ``long`` is ``int64``, ``double`` is float64.
EVENTS_DTYPES = {
    "drive_id": "object",
    "event_type": "object",
    "start_frame": "int64",
    "end_frame": "int64",
    "start_ts": "float64",
    "end_ts": "float64",
    "params": "object",
}
DRIVES_DTYPES = {
    "drive_id": "object",
    "vehicle_id": "object",
    "vehicle_kind": "object",
    "drive_start_utc": "object",
    "start_ts": "float64",
    "duration_s": "float64",
    "n_frames": "int64",
    "distance_m": "float64",
    "n_events": "int64",
    "n_errors": "int64",
}
