"""Shared helpers for the Spark ingest tests: raw DataFrames, files, read-back and comparison."""

import json
from pathlib import Path
from typing import Any

import pytest
from pyspark.sql import DataFrame, SparkSession

from fixtures.message_loader import ValidCase
from vl.schemas import RAW_FRAMES_COLUMNS

Row = tuple[Any, ...]


def raw_df(spark: SparkSession, lines: list[str]) -> DataFrame:
    """Build the one-column ``value`` DataFrame that ``parse_messages`` takes.

    Args:
        spark: The shared Spark session.
        lines: Raw message strings, one per row.

    Returns:
        A DataFrame with a single string column ``value``.
    """
    return spark.createDataFrame([(line,) for line in lines], "value string")


def ddl_dtypes(spark: SparkSession, ddl: str) -> list[tuple[str, str]]:
    """Get the (name, type) pairs Spark itself reads from a DDL string.

    Args:
        spark: The shared Spark session.
        ddl: A DDL string from ``vl.schemas``.

    Returns:
        ``DataFrame.dtypes`` of an empty DataFrame created from the DDL.
    """
    return spark.createDataFrame([], ddl).dtypes


def message(drive_id: str, frame_idx: int, points: list[list[int]], **extra: object) -> str:
    """Build one valid message line.

    Args:
        drive_id: Drive id.
        frame_idx: Frame index.
        points: Integer-cm points.
        **extra: Additional fields, for example ``vehicle_id``.

    Returns:
        The message as a JSON line, with ``ts = frame_idx * 0.0625``.
    """
    fields: dict[str, Any] = {"drive_id": drive_id, "frame_idx": frame_idx, "points": points}
    fields.update(ts=frame_idx * 0.0625, **extra)
    return json.dumps(fields)


def write_lines(directory: Path, name: str, lines: list[str]) -> None:
    """Write message lines to a file, creating the directory.

    Args:
        directory: Target directory.
        name: File name.
        lines: One message per line.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text("\n".join(lines) + "\n")


def frame_key(row: Row) -> tuple[str, int, bool]:
    """Sort key of a ``raw_frames`` tuple: (drive_id, frame_idx, has vehicle_id).

    Args:
        row: A tuple in ``RAW_FRAMES_COLUMNS`` order.

    Returns:
        The key that makes the order of read-back rows deterministic.
    """
    return (row[0], row[4], row[1] is not None)


def parquet_files(out_dir: Path) -> set[str]:
    """List the Parquet files under a directory.

    Args:
        out_dir: Warehouse directory.

    Returns:
        Paths relative to ``out_dir``.
    """
    return {str(p.relative_to(out_dir)) for p in out_dir.rglob("*.parquet")}


def read_table(spark: SparkSession, out_dir: Path, table: str) -> DataFrame | None:
    """Read an output table back, or None when it was not written.

    Args:
        spark: The shared Spark session.
        out_dir: Warehouse directory.
        table: ``raw_frames`` or ``rejects``.

    Returns:
        The table, or None when its directory or its Parquet files are missing.
    """
    path = out_dir / table
    if not list(path.rglob("*.parquet")):
        return None
    return spark.read.parquet(str(path))


def table_rows(spark: SparkSession, out_dir: Path, table: str) -> list[Row]:
    """Read an output table as sorted tuples (empty list when it is absent).

    ``raw_frames`` is read back with the columns in ``RAW_FRAMES_COLUMNS`` order, because
    the partition column ``drive_id`` is placed last by Parquet partition discovery.

    Args:
        spark: The shared Spark session.
        out_dir: Warehouse directory.
        table: ``raw_frames`` or ``rejects``.

    Returns:
        Rows sorted by (drive_id, frame_idx, has vehicle_id) for ``raw_frames`` and by
        (payload, ingested_at) for ``rejects``, so comparisons ignore file order.
    """
    df = read_table(spark, out_dir, table)
    if df is None:
        return []
    if table == "rejects":
        return sorted((tuple(r) for r in df.collect()), key=lambda row: (row[0], row[2]))
    rows = [tuple(r) for r in df.select(*RAW_FRAMES_COLUMNS).collect()]
    return sorted(rows, key=frame_key)


def case_row(case: ValidCase) -> Row:
    """Turn a valid fixture case into a ``raw_frames`` tuple.

    Args:
        case: A valid case from ``expected.json``.

    Returns:
        The tuple in ``RAW_FRAMES_COLUMNS`` order.
    """
    return (
        case.drive_id,
        case.vehicle_id,
        case.vehicle_kind,
        case.drive_start_utc,
        case.frame_idx,
        case.ts,
        case.n_points,
        case.x,
        case.y,
        case.z,
    )


def assert_row_matches(actual: Row, expected: Row, label: str = "") -> None:
    """Compare a ``raw_frames`` tuple with its expectation.

    Everything is compared exactly except the centroid, which uses ``pytest.approx``.

    Args:
        actual: Row read from Spark.
        expected: Row derived by hand (see `case_row`).
        label: Case name, put in every assertion message.
    """
    assert actual[:7] == expected[:7], label
    for got, want in zip(actual[7:], expected[7:], strict=True):
        assert (got is None) == (want is None), label
        if want is not None:
            assert got == pytest.approx(want, abs=1e-9), label


def assert_rows_match(actual: list[Row], cases: list[ValidCase]) -> None:
    """Compare read-back ``raw_frames`` rows with fixture cases, ignoring order.

    Args:
        actual: Rows read back from the output table.
        cases: The valid cases that were fed in.
    """
    expected = sorted((case_row(c) for c in cases), key=frame_key)
    assert len(actual) == len(expected)
    for got, want in zip(sorted(actual, key=frame_key), expected, strict=True):
        assert_row_matches(got, want)
