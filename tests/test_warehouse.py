"""Tests for the DuckDB query layer (``vl.warehouse``).

Warehouses are built by ``run_transform`` over the golden fixtures (the expectations come from
the golden ``expected`` keys) or written directly as Parquet; empty warehouses need no Spark.
"""

from pathlib import Path

import pandas as pd
import pytest

from fixtures.loader import list_golden_cases, load_golden
from pipeline_helpers import frames_table
from transform_helpers import DDLS, GoldenOutput, ddl_duckdb_columns
from vl.schemas import DRIVES_COLUMNS
from vl.warehouse import EXAMPLE_QUERIES, Warehouse

VIEWS = ["drives", "events", "frames", "raw_frames", "rejects"]
# The documented columns of the example queries, in order.
QUERY_COLUMNS = {
    "hard_brakes_per_drive": ["drive_id", "n_hard_brakes"],
    "events_by_type": ["event_type", "n_events"],
    "error_counts": ["error_code", "n_frames"],
    "drives_overview": DRIVES_COLUMNS,
    "fastest_drives": ["drive_id", "vehicle_id", "max_speed_ms"],
}
PRESENT = [n for n in list_golden_cases() if load_golden(n).expected["n_frames_out"] > 0]


@pytest.fixture(scope="module")
def golden_wh(golden_output: GoldenOutput) -> Warehouse:
    """A warehouse over the transform output of all golden cases."""
    return Warehouse(str(golden_output.warehouse))


@pytest.fixture
def empty_wh(tmp_path: Path) -> Warehouse:
    """A warehouse over an empty directory."""
    return Warehouse(str(tmp_path))


def golden_events(event_type: str) -> dict[str, int]:
    """Count the golden events of one type per drive (drives without one are absent)."""
    counts = {}
    for name in PRESENT:
        n = sum(e["event_type"] == event_type for e in load_golden(name).expected["events"])
        if n:
            counts[name] = n
    return counts


# --- structure -----------------------------------------------------------------------------


def test_missing_tables_are_typed_empty_views(empty_wh: Warehouse) -> None:
    assert sorted(empty_wh.tables()) == VIEWS
    for table in DDLS:
        assert empty_wh.query(f"select count(*) c from {table}")["c"][0] == 0, table
        described = empty_wh.query(f"describe {table}")
        columns = list(zip(described["column_name"], described["column_type"]))
        assert columns == ddl_duckdb_columns(DDLS[table]), table


def test_golden_warehouse_views_and_hive_partition_column(golden_wh: Warehouse) -> None:
    assert sorted(golden_wh.tables()) == VIEWS
    drives = golden_wh.query("select distinct drive_id from frames order by drive_id")
    assert drives["drive_id"].tolist() == sorted(PRESENT)


def test_params_are_bound_not_interpolated(empty_wh: Warehouse) -> None:
    tricky = "x'; drop view frames; --"
    assert empty_wh.query("select ? as v", [tricky])["v"][0] == tricky
    found = empty_wh.query("select count(*) c from frames where drive_id = ?", [tricky])
    assert found["c"][0] == 0
    assert empty_wh.query("select count(*) c from frames")["c"][0] == 0  # the view survived


def write_frames(directory: Path, ids: list[str]) -> None:
    """Write a hive-partitioned ``frames`` table with 3 hand-made rows per drive id."""
    table = pd.concat([frames_table(3, drive_id=i) for i in ids])
    table.to_parquet(directory / "frames", partition_cols=["drive_id"])


# "007" guards the pyarrow side; the all-numeric ids guard the DuckDB type inference (DuckDB keeps
# VARCHAR only when an id has a leading zero, otherwise it would infer BIGINT).
def test_drive_id_stays_a_string(tmp_path: Path) -> None:
    for number, ids in enumerate((["007", "12"], ["12", "13"])):
        directory = tmp_path / str(number)
        directory.mkdir()
        write_frames(directory, ids)
        found = Warehouse(str(directory)).query(
            "select distinct drive_id, typeof(drive_id) t from frames order by drive_id"
        )
        assert found["drive_id"].tolist() == ids, ids
        assert set(found["t"]) == {"VARCHAR"}, ids


def test_files_left_by_a_crashed_spark_write_are_not_read(tmp_path: Path) -> None:
    """Spark stages a write under ``<table>/_temporary/``; a crash leaves the files there."""
    write_frames(tmp_path, ["live"])
    leftover = tmp_path / "frames" / "_temporary" / "0" / "_temporary" / "attempt_0"
    write_frames(leftover, ["ghost"])  # an uncommitted copy, laid out as Spark stages it
    (tmp_path / "events" / "_temporary").mkdir(parents=True)  # a flat table with only leftovers
    frames_table(1, drive_id="ghost").to_parquet(tmp_path / "events" / "_temporary" / "p.parquet")
    wh = Warehouse(str(tmp_path))
    assert wh.query("select distinct drive_id from frames")["drive_id"].tolist() == ["live"]
    assert wh.query("select count(*) n from events")["n"][0] == 0


def test_creating_a_warehouse_writes_no_database_file(tmp_path: Path) -> None:
    write_frames(tmp_path, ["a"])
    before = sorted(p.name for p in tmp_path.rglob("*"))
    warehouse = Warehouse(str(tmp_path))
    warehouse.query("select count(*) c from frames")
    assert sorted(p.name for p in tmp_path.rglob("*")) == before
    assert not list(tmp_path.glob("*.duckdb")) and not list(tmp_path.glob("*.db"))


# --- example queries -----------------------------------------------------------------------


def test_example_queries_run_and_make_sense(golden_wh: Warehouse, empty_wh: Warehouse) -> None:
    assert sorted(EXAMPLE_QUERIES) == sorted(QUERY_COLUMNS)
    for name, columns in QUERY_COLUMNS.items():
        result = golden_wh.query(EXAMPLE_QUERIES[name])
        assert isinstance(result, pd.DataFrame), name
        assert list(result.columns) == columns, name
        assert len(result) > 0, name
        on_empty = empty_wh.query(EXAMPLE_QUERIES[name])
        assert list(on_empty.columns) == columns, name
        assert on_empty.empty, name


def test_error_counts_match_the_golden_codes(golden_wh: Warehouse) -> None:
    expected: dict[str, int] = {}
    for name in PRESENT:
        for code in load_golden(name).expected["error_codes"].values():
            expected[code] = expected.get(code, 0) + 1
    assert expected == {"VL-001": 2, "VL-002": 1, "VL-004": 2, "VL-006": 1}  # g05+g10, ...
    counts = golden_wh.query(EXAMPLE_QUERIES["error_counts"])
    assert dict(zip(counts["error_code"], counts["n_frames"])) == expected
    assert counts["error_code"].tolist() == sorted(expected)


def test_hard_brakes_per_drive_match_the_golden_events(golden_wh: Warehouse) -> None:
    brakes = golden_events("HARD_BRAKE")
    assert brakes == {"g02_hard_brake": 1, "g10_gap_inside_brake": 2}
    result = golden_wh.query(EXAMPLE_QUERIES["hard_brakes_per_drive"])
    assert result["drive_id"].tolist() == PRESENT  # every drive, ordered, zero when none
    assert dict(zip(result["drive_id"], result["n_hard_brakes"])) == {
        name: brakes.get(name, 0) for name in PRESENT
    }


def test_events_by_type_match_the_golden_events(golden_wh: Warehouse) -> None:
    result = golden_wh.query(EXAMPLE_QUERIES["events_by_type"])
    expected = {
        kind: sum(golden_events(kind).values()) for kind in ("HARD_BRAKE", "SHARP_TURN", "STOP")
    }
    assert expected == {"HARD_BRAKE": 3, "SHARP_TURN": 1, "STOP": 1}
    assert dict(zip(result["event_type"], result["n_events"])) == expected
    assert result["event_type"].tolist() == sorted(expected)


def test_drives_overview_is_one_row_per_drive(golden_wh: Warehouse) -> None:
    result = golden_wh.query(EXAMPLE_QUERIES["drives_overview"])
    assert result["drive_id"].tolist() == PRESENT
    frames_out = {n: load_golden(n).expected["n_frames_out"] for n in PRESENT}
    assert dict(zip(result["drive_id"], result["n_frames"])) == frames_out


def test_fastest_drives_are_the_five_highest_top_speeds(golden_wh: Warehouse) -> None:
    result = golden_wh.query(EXAMPLE_QUERIES["fastest_drives"])
    speeds = result["max_speed_ms"].tolist()
    assert len(result) == 5
    assert speeds == sorted(speeds, reverse=True)
    # g02 and g10 cruise at 15 m/s; six drives (g01, g05 to g09) reach about 10 m/s, two (g03,
    # g04) 5 m/s and g11 has no speed: the other three places go to the 10 m/s group.
    assert set(result["drive_id"][:2]) == {"g02_hard_brake", "g10_gap_inside_brake"}
    assert speeds[:2] == pytest.approx([15.0, 15.0], abs=0.5)
    assert speeds[2:] == pytest.approx([10.0] * 3, abs=0.5)


def test_fastest_drives_report_the_vehicle(tmp_path: Path) -> None:
    # The golden drives carry no metadata (null vehicle_id), so this uses hand-made frames.
    for drive_id, vehicle, speed in [("slow", "veh-A", 5.0), ("quick", "veh-B", 12.0)]:
        table = frames_table(3, drive_id=drive_id, vehicle_id=vehicle, speed=speed)
        table.to_parquet(tmp_path / "frames", partition_cols=["drive_id"])
    result = Warehouse(str(tmp_path)).query(EXAMPLE_QUERIES["fastest_drives"])
    assert result.values.tolist() == [["quick", "veh-B", 12.0], ["slow", "veh-A", 5.0]]
