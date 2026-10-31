"""Shared helpers for the transform and warehouse tests: raw rows, warehouses, read-back."""

from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from fixtures.loader import GoldenCase, list_golden_cases, load_golden
from vl.config import DEFAULT, Config
from vl.etl.transform import run_transform
from vl.mock.drive_sim import SimConfig
from vl.schemas import (
    DRIVES_COLUMNS,
    DRIVES_DDL,
    EVENTS_COLUMNS,
    EVENTS_DDL,
    FRAMES_COLUMNS,
    FRAMES_DDL,
    RAW_FRAMES_COLUMNS,
    RAW_FRAMES_DDL,
    REJECTS_DDL,
)

RawRow = tuple[Any, ...]
CLEAN = SimConfig.clean()
METADATA = ["vehicle_id", "vehicle_kind", "drive_start_utc"]
KINEMATICS = ["vx", "vy", "speed", "accel", "yaw_rate"]
READ_COLUMNS = {"frames": FRAMES_COLUMNS, "events": EVENTS_COLUMNS, "drives": DRIVES_COLUMNS}
TABLES = ("frames", "events", "drives")
DDLS = {
    "frames": FRAMES_DDL,
    "events": EVENTS_DDL,
    "drives": DRIVES_DDL,
    "rejects": REJECTS_DDL,
    "raw_frames": RAW_FRAMES_DDL,
}
# DuckDB type of every Spark type used by the DDL strings.
DUCKDB_TYPES = {
    "string": "VARCHAR",
    "long": "BIGINT",
    "double": "DOUBLE",
    "int": "INTEGER",
    "timestamp": "TIMESTAMP",
}
GOLDEN_FIELDS = ("frame_idx", "ts", "n_points", "x", "y", "z")
SORT_KEYS = {
    "frames": ["drive_id", "frame_idx"],
    "events": ["drive_id", "start_frame", "event_type"],
    "drives": ["drive_id"],
}


def flagged_codes(frames: pd.DataFrame) -> dict[str, str]:
    """Map ``str(frame_idx)`` to the error code of every flagged frame."""
    flagged = frames[frames["error_code"].notna()]
    return {str(i): str(c) for i, c in zip(flagged["frame_idx"], flagged["error_code"])}


def frame_at(frames: pd.DataFrame, frame_idx: int) -> "pd.Series[Any]":
    """Return the single row of ``frames`` with the given ``frame_idx``."""
    rows = frames[frames["frame_idx"] == frame_idx]
    assert len(rows) == 1, f"expected exactly one row for frame {frame_idx}"
    return rows.iloc[0]


def check_events(events: pd.DataFrame, expected: list[dict[str, Any]]) -> None:
    """Assert the detected events match the expected ones within each ``frame_tol``."""
    found = events.sort_values("start_frame").reset_index(drop=True)
    assert len(found) == len(expected), f"events found:\n{found}"
    for row, want in zip(found.itertuples(), expected):
        assert row.event_type == want["event_type"]
        assert abs(row.start_frame - want["start_frame"]) <= want["frame_tol"]
        assert abs(row.end_frame - want["end_frame"]) <= want["frame_tol"]


def check_speeds(frames: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    """Assert the speed at each checked frame."""
    for check in checks:
        speed = frame_at(frames, check["frame_idx"])["speed"]
        assert speed == pytest.approx(check["speed"], abs=check["tol"]), check


def check_yaw_rates(frames: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    """Assert the yaw rate at each checked frame."""
    for check in checks:
        yaw = frame_at(frames, check["frame_idx"])["yaw_rate"]
        assert yaw == pytest.approx(check["yaw_rate"], abs=check["tol"]), check


def check_null_kinematics(frames: pd.DataFrame, frame_ids: list[int]) -> None:
    """Assert suspect frames have no velocity, speed, acceleration and yaw rate."""
    for frame_idx in frame_ids:
        row = frame_at(frames, frame_idx)
        for column in KINEMATICS:
            assert pd.isna(row[column]), f"{column} of suspect frame {frame_idx}"


def check_no_event_contains(events: pd.DataFrame, frame_ids: list[int]) -> None:
    """Assert no event covers any of the given frames (both ends inclusive)."""
    for frame_idx in frame_ids:
        inside = (events["start_frame"] <= frame_idx) & (frame_idx <= events["end_frame"])
        assert not inside.any(), f"an event contains frame {frame_idx}:\n{events[inside]}"


def golden_rows(case: GoldenCase) -> list[RawRow]:
    """Turn a golden case into ``raw_frames`` tuples under the drive id equal to the case name.

    The three metadata columns are null, as for messages that carry no metadata.

    Args:
        case: The golden case (an empty case gives no rows).

    Returns:
        Tuples in ``RAW_FRAMES_COLUMNS`` order.
    """
    return [(case.case, None, None, None, *(r[key] for key in GOLDEN_FIELDS)) for r in case.raw]


def frame_rows(raw: pd.DataFrame) -> list[RawRow]:
    """Turn a raw DataFrame (``RAW_FRAMES_COLUMNS``) into tuples of Python values, NaN as None.

    Args:
        raw: A DataFrame such as ``raw_from_frames`` returns.

    Returns:
        Tuples in ``RAW_FRAMES_COLUMNS`` order.
    """
    cells = raw[RAW_FRAMES_COLUMNS].astype(object).where(raw[RAW_FRAMES_COLUMNS].notna(), None)
    return [tuple(row) for row in cells.values.tolist()]


def write_raw_frames(
    spark: SparkSession, warehouse: Path, rows: list[RawRow], mode: str = "overwrite"
) -> None:
    """Write rows as the ``raw_frames`` Parquet table (partitioned by ``drive_id``).

    Args:
        spark: The shared Spark session.
        warehouse: Warehouse directory.
        rows: Tuples in ``RAW_FRAMES_COLUMNS`` order.
        mode: Spark save mode (``overwrite`` or ``append``).
    """
    frame = spark.createDataFrame(rows, RAW_FRAMES_DDL)
    frame.write.mode(mode).partitionBy("drive_id").parquet(str(warehouse / "raw_frames"))


def all_golden_rows() -> list[RawRow]:
    """Collect the raw rows of every golden case (the empty case contributes none).

    Returns:
        The rows of all cases.
    """
    return [row for name in list_golden_cases() for row in golden_rows(load_golden(name))]


def build_golden_warehouse(spark: SparkSession, directory: Path) -> tuple[Path, dict[str, int]]:
    """Write every golden case as ``raw_frames`` and run the transform once for all of them.

    Args:
        spark: The shared Spark session.
        directory: Empty directory that becomes the warehouse.

    Returns:
        The warehouse directory and the row counts ``run_transform`` returned.
    """
    write_raw_frames(spark, directory, all_golden_rows())
    return directory, run_transform(spark, str(directory), DEFAULT)


def transform_rows(
    spark: SparkSession, directory: Path, rows: list[RawRow], cfg: Config = DEFAULT
) -> dict[str, int]:
    """Write rows as ``raw_frames`` in ``directory`` and run the transform.

    Args:
        spark: The shared Spark session.
        directory: Warehouse directory.
        rows: Raw tuples.
        cfg: Pipeline thresholds.

    Returns:
        The row counts ``run_transform`` returned.
    """
    write_raw_frames(spark, directory, rows)
    return run_transform(spark, str(directory), cfg)


def read_pd(spark: SparkSession, warehouse: Path, table: str) -> pd.DataFrame:
    """Read an output table with Spark as a pandas DataFrame in a deterministic order.

    The columns are in the table's documented order (Parquet partition discovery puts
    ``drive_id`` last), the rows sorted by the table's key.

    Args:
        spark: The shared Spark session.
        warehouse: Warehouse directory.
        table: ``frames``, ``events`` or ``drives``.

    Returns:
        The table content, positional index.
    """
    frame = spark.read.parquet(str(warehouse / table)).select(*READ_COLUMNS[table])
    return frame.toPandas().sort_values(SORT_KEYS[table]).reset_index(drop=True)


def of_drive(table: pd.DataFrame, drive_id: str) -> pd.DataFrame:
    """Select the rows of one drive.

    Args:
        table: A ``frames``, ``events`` or ``drives`` DataFrame.
        drive_id: The drive id.

    Returns:
        The rows of that drive with a positional index.
    """
    return table[table["drive_id"] == drive_id].reset_index(drop=True)


def ddl_duckdb_columns(ddl: str) -> list[tuple[str, str]]:
    """Parse a Spark DDL string into (column, DuckDB type) pairs.

    Args:
        ddl: A DDL string from ``vl.schemas``.

    Returns:
        The columns in DDL order with the DuckDB type name of each.
    """
    pairs = (column.split() for column in ddl.split(", "))
    return [(name, DUCKDB_TYPES[spark_type]) for name, spark_type in pairs]


class GoldenOutput(NamedTuple):
    """The golden warehouse: its directory, the returned counts and the tables read back."""

    warehouse: Path
    counts: dict[str, int]
    frames: pd.DataFrame
    events: pd.DataFrame
    drives: pd.DataFrame


@pytest.fixture(scope="session")
def golden_output(spark: SparkSession, tmp_path_factory: pytest.TempPathFactory) -> GoldenOutput:
    """Run the transform once per test session over all golden cases written as ``raw_frames``.

    Registered as a plugin by ``conftest.py`` so ``test_transform`` and ``test_warehouse`` share
    the one build.
    """
    warehouse, counts = build_golden_warehouse(spark, tmp_path_factory.mktemp("golden"))
    tables = [read_pd(spark, warehouse, table) for table in TABLES]
    return GoldenOutput(warehouse, counts, *tables)
