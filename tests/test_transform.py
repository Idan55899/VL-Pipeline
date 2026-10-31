"""Tests for the Spark batch transform (``vl.etl.transform.run_transform``).

The golden cases are written as ``raw_frames`` Parquet and checked against each case's
``expected`` keys, an oracle independent of the transform code; the other expectations are
hand-derived in the tests.
"""

import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal
from pyspark.sql import SparkSession

from fixtures.loader import GoldenCase, list_golden_cases, load_golden
from ingest_helpers import ddl_dtypes
from pipeline_helpers import raw_from_frames
from transform_helpers import (
    CLEAN,
    DDLS,
    METADATA,
    TABLES,
    GoldenOutput,
    RawRow,
    check_events,
    check_no_event_contains,
    check_null_kinematics,
    check_speeds,
    check_yaw_rates,
    ddl_duckdb_columns,
    flagged_codes,
    frame_rows,
    golden_rows,
    of_drive,
    read_pd,
    transform_rows,
    write_raw_frames,
)
from vl.config import DEFAULT
from vl.etl.transform import run_transform
from vl.mock.drive_sim import simulate_drive
from vl.mock.faults import Fault
from vl.schemas import FRAMES_COLUMNS, FRAMES_DDL
from vl.warehouse import Warehouse

G01 = "g01_constant_speed"
G02 = "g02_hard_brake"


def read_all(spark: SparkSession, warehouse: Path) -> dict[str, pd.DataFrame]:
    """Read the three output tables of a warehouse."""
    return {table: read_pd(spark, warehouse, table) for table in TABLES}


def run_rows(spark: SparkSession, directory: Path, rows: list[RawRow]) -> dict[str, pd.DataFrame]:
    """Transform raw rows in ``directory`` and read the three tables back."""
    transform_rows(spark, directory, rows)
    return read_all(spark, directory)


def check_drive_row(drives: pd.DataFrame, case: GoldenCase) -> None:
    """Assert the ``drives`` rows of a case: none for an empty case, else the expected values.

    ``start_ts`` and ``duration_s`` are derived from the raw rows with plain arithmetic: the
    first ``ts`` and the last ``ts`` minus the first, in ``frame_idx`` order.
    """
    expected = case.expected
    if expected["n_frames_out"] == 0:
        assert drives.empty  # an empty case contributes no rows, hence no drive
        return
    assert len(drives) == 1
    row = drives.iloc[0]
    ordered = sorted(case.raw, key=lambda r: r["frame_idx"])
    assert row["start_ts"] == pytest.approx(ordered[0]["ts"], abs=1e-9)
    assert row["duration_s"] == pytest.approx(ordered[-1]["ts"] - ordered[0]["ts"], abs=1e-9)
    assert row["n_frames"] == expected["n_frames_out"]
    assert row["n_events"] == len(expected["events"])
    assert row["n_errors"] == len(expected["error_codes"])
    if "distance_m" in expected:
        assert row["distance_m"] == pytest.approx(
            expected["distance_m"], abs=expected["distance_tol"]
        )


# --- golden fixtures through Spark ---------------------------------------------------------


def check_golden_case(golden_output: GoldenOutput, name: str) -> None:
    """Assert the transform output of one golden case against its ``expected`` keys."""
    case = load_golden(name)
    expected = case.expected
    frames = of_drive(golden_output.frames, name)
    events = of_drive(golden_output.events, name)
    assert list(frames.columns) == FRAMES_COLUMNS
    assert len(frames) == expected["n_frames_out"]
    assert flagged_codes(frames) == expected["error_codes"]
    check_events(events, expected["events"])
    check_no_event_contains(events, expected.get("no_event_contains_frames", []))
    check_speeds(frames, expected.get("speed_checks", []))
    check_yaw_rates(frames, expected.get("yaw_rate_checks", []))
    check_null_kinematics(frames, expected.get("null_kinematics_frames", []))
    check_drive_row(of_drive(golden_output.drives, name), case)


def test_golden_cases_through_spark(golden_output: GoldenOutput) -> None:
    """All golden cases come out of one transform run with their expected values."""
    for name in list_golden_cases():
        try:
            check_golden_case(golden_output, name)
        except AssertionError as error:
            raise AssertionError(f"golden case {name}: {error}") from error
    cases = [load_golden(name).expected for name in list_golden_cases()]
    assert golden_output.counts == {
        "frames": sum(e["n_frames_out"] for e in cases),
        "events": sum(len(e["events"]) for e in cases),
        "drives": sum(1 for e in cases if e["n_frames_out"] > 0),
    }
    assert golden_output.counts == {t: len(getattr(golden_output, t)) for t in TABLES}


# --- schemas -------------------------------------------------------------------------------


def check_frames_layout(spark: SparkSession, frames_dir: Path) -> None:
    """Assert the ``frames`` types, the ``drive_id=...`` partition directories and file columns."""
    frames = spark.read.parquet(str(frames_dir))
    assert frames.select(*FRAMES_COLUMNS).dtypes == ddl_dtypes(spark, FRAMES_DDL)
    assert sorted(frames.columns) == sorted(FRAMES_COLUMNS)  # no extra or missing column
    present = [n for n in list_golden_cases() if golden_rows(load_golden(n))]
    assert sorted(p.name for p in frames_dir.iterdir() if p.is_dir()) == sorted(
        f"drive_id={name}" for name in present
    )
    first_file = next(frames_dir.rglob("*.parquet"))
    assert pq.read_schema(first_file).names == [c for c in FRAMES_COLUMNS if c != "drive_id"]


def test_output_schemas_match_ddls(spark: SparkSession, golden_output: GoldenOutput) -> None:
    check_frames_layout(spark, golden_output.warehouse / "frames")
    for table in ("events", "drives"):
        written = spark.read.parquet(str(golden_output.warehouse / table))
        assert written.dtypes == ddl_dtypes(spark, DDLS[table])  # names, order and types


# --- reruns, replays and empty inputs ------------------------------------------------------


def test_idempotent(spark: SparkSession, tmp_path: Path) -> None:
    rows = [r for name in (G02, "g06_position_jump") for r in golden_rows(load_golden(name))]
    write_raw_frames(spark, tmp_path, rows)
    first_counts = run_transform(spark, str(tmp_path), DEFAULT)
    first = read_all(spark, tmp_path)
    assert run_transform(spark, str(tmp_path), DEFAULT) == first_counts
    second = read_all(spark, tmp_path)
    for table in TABLES:
        assert_frame_equal(second[table], first[table])
    assert first_counts["events"] == 1  # the brake of g02, nothing duplicated by the rerun


def test_replayed_duplicates_not_duplicated(spark: SparkSession, tmp_path: Path) -> None:
    rows = golden_rows(load_golden(G01))
    once = run_rows(spark, tmp_path / "once", rows)["frames"]
    write_raw_frames(spark, tmp_path / "twice", rows, mode="append")
    write_raw_frames(spark, tmp_path / "twice", rows, mode="append")
    run_transform(spark, str(tmp_path / "twice"), DEFAULT)
    twice = read_pd(spark, tmp_path / "twice", "frames")
    assert once["error_code"].isna().all()
    assert len(twice) == len(once) == 60
    assert twice["frame_idx"].is_unique
    positional = ["drive_id", "frame_idx", "ts", "x", "y", "z", *METADATA]
    assert_frame_equal(twice[positional], once[positional])
    assert (twice["error_code"] == "VL-004").all()  # every replayed frame, flagged once


def prepare_empty(spark: SparkSession, warehouse: Path, kind: str) -> None:
    """Create the no-data inputs: no ``raw_frames`` at all, an empty directory, an empty table."""
    warehouse.mkdir(parents=True)
    if kind == "empty_dir":
        (warehouse / "raw_frames").mkdir()
    if kind == "empty_table":
        write_raw_frames(spark, warehouse, [])


def count_rows(spark: SparkSession, warehouse: Path, table: str) -> int:
    """Count the rows of an output table, read with its DDL (works for a table without files)."""
    return spark.read.schema(DDLS[table]).parquet(str(warehouse / table)).count()


def check_empty_outputs(spark: SparkSession, warehouse: Path) -> None:
    """Assert typed, empty output tables and typed empty Warehouse views."""
    for table in TABLES:  # no rows; whether an empty frames table has a schema file is open
        assert count_rows(spark, warehouse, table) == 0, table
    for table in ("events", "drives"):
        written = spark.read.parquet(str(warehouse / table))
        assert written.dtypes == ddl_dtypes(spark, DDLS[table]), table
    view = Warehouse(str(warehouse))
    for table in TABLES:
        described = view.query(f"describe {table}")
        assert view.query(f"select count(*) c from {table}")["c"][0] == 0, table
        columns = list(zip(described["column_name"], described["column_type"]))
        assert columns == ddl_duckdb_columns(DDLS[table]), table


def test_empty_raw_frames_gives_typed_empty_outputs(spark: SparkSession, tmp_path: Path) -> None:
    for kind in ("missing", "empty_dir", "empty_table"):
        warehouse = tmp_path / kind
        prepare_empty(spark, warehouse, kind)
        counts = run_transform(spark, str(warehouse), DEFAULT)
        assert counts == {"frames": 0, "events": 0, "drives": 0}, kind
        try:
            check_empty_outputs(spark, warehouse)
        except AssertionError as error:
            raise AssertionError(f"raw_frames {kind}: {error}") from error


def stale_partitions(frames_dir: Path) -> list[str]:
    """List the ``drive_id=...`` partition directories of a ``frames`` table."""
    return sorted(p.name for p in frames_dir.glob("drive_id=*"))


def test_rerun_after_drives_were_removed_drops_them(spark: SparkSession, tmp_path: Path) -> None:
    both = golden_rows(load_golden(G02)) + golden_rows(load_golden("g06_position_jump"))
    transform_rows(spark, tmp_path, both)
    assert len(stale_partitions(tmp_path / "frames")) == 2
    tables = run_rows(spark, tmp_path, golden_rows(load_golden(G02)))  # raw_frames overwritten
    assert stale_partitions(tmp_path / "frames") == [f"drive_id={G02}"]  # no stale partition
    for table in TABLES:
        assert set(tables[table]["drive_id"]) == {G02}
    assert len(tables["drives"]) == 1


def test_rerun_after_raw_frames_vanished_clears_the_outputs(
    spark: SparkSession, tmp_path: Path
) -> None:
    for how in ("removed", "emptied"):
        warehouse = tmp_path / how
        transform_rows(spark, warehouse, golden_rows(load_golden(G02)))
        assert run_transform(spark, str(warehouse), DEFAULT)["frames"] == 160, how
        if how == "removed":
            shutil.rmtree(warehouse / "raw_frames")
        else:
            write_raw_frames(spark, warehouse, [])
        counts = run_transform(spark, str(warehouse), DEFAULT)
        assert counts == {"frames": 0, "events": 0, "drives": 0}, how
        assert stale_partitions(warehouse / "frames") == [], how
        for table in TABLES:
            assert count_rows(spark, warehouse, table) == 0, f"{how}: {table}"


# --- metadata, independence, configuration, empty clouds -----------------------------------


def test_metadata_columns_flow_to_drives(spark: SparkSession, tmp_path: Path) -> None:
    drives = [
        simulate_drive("m-1", vehicle_kind="suv", n_frames=120, seed=1, sim=CLEAN),
        simulate_drive("m-2", vehicle_kind="truck", n_frames=120, seed=2, sim=CLEAN),
    ]
    rows = [r for drive in drives for r in frame_rows(raw_from_frames(drive.frames))]
    tables = run_rows(spark, tmp_path, rows)
    for drive in drives:
        message = drive.frames[0]
        summary = of_drive(tables["drives"], message["drive_id"]).iloc[0]
        frames = of_drive(tables["frames"], message["drive_id"])
        for column in METADATA:
            assert summary[column] == message[column]
            assert (frames[column] == message[column]).all()


def test_two_drives_are_independent(spark: SparkSession, tmp_path: Path) -> None:
    first = simulate_drive("ind-a", n_frames=300, sim=CLEAN, faults=(Fault("drop", 100),))
    second = simulate_drive("ind-b", profile="city", vehicle_kind="truck", n_frames=300, seed=5)
    rows_a = frame_rows(raw_from_frames(first.frames))
    alone = run_rows(spark, tmp_path / "alone", rows_a)
    both = run_rows(spark, tmp_path / "both", rows_a + frame_rows(raw_from_frames(second.frames)))
    assert sorted(both["drives"]["drive_id"]) == ["ind-a", "ind-b"]
    for table in TABLES:
        assert_frame_equal(of_drive(both[table], "ind-a"), of_drive(alone[table], "ind-a"))


def test_cfg_is_applied(spark: SparkSession, tmp_path: Path) -> None:
    rows = golden_rows(load_golden(G01))
    write_raw_frames(spark, tmp_path, rows)
    run_transform(spark, str(tmp_path), DEFAULT)
    assert flagged_codes(read_pd(spark, tmp_path, "frames")) == {}
    run_transform(spark, str(tmp_path), replace(DEFAULT, max_jump_m=0.1))
    # Every step is 0.625 m (10 m/s * 0.0625 s) > 0.1 m: all frames after the first jump.
    codes = {str(i): "VL-002" for i in range(1, 60)}
    assert flagged_codes(read_pd(spark, tmp_path, "frames")) == codes


def test_cfg_reaches_events_and_drives(spark: SparkSession, tmp_path: Path) -> None:
    rows = golden_rows(load_golden(G02))
    write_raw_frames(spark, tmp_path, rows)
    run_transform(spark, str(tmp_path), DEFAULT)
    assert len(read_pd(spark, tmp_path, "events")) == 1  # the HARD_BRAKE of g02
    # The true deceleration is -5 m/s^2, so a -6 m/s^2 threshold finds no hard brake.
    counts = run_transform(spark, str(tmp_path), replace(DEFAULT, hard_brake_ms2=-6.0))
    assert counts["events"] == 0
    assert read_pd(spark, tmp_path, "events").empty
    assert read_pd(spark, tmp_path, "drives")["n_events"].tolist() == [0]


def test_empty_cloud_rows_flow_through(spark: SparkSession, tmp_path: Path) -> None:
    rows = golden_rows(load_golden(G01))
    ts25 = rows[25][5]
    rows[25] = (G01, None, None, None, 25, ts25, 0, None, None, None)  # n_points 0, no centroid
    frames = of_drive(run_rows(spark, tmp_path, rows)["frames"], G01)
    assert len(frames) == 60
    assert flagged_codes(frames) == {"25": "VL-006"}
    check_null_kinematics(frames, [25])
    assert frames.loc[25, ["x", "y", "z"]].isna().all()
    check_speeds(
        frames, [{"frame_idx": i, "speed": 10.0, "tol": 0.3} for i in (23, 24, 26, 27, 30)]
    )
