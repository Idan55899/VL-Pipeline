"""Phase 2 sanity test: one end-to-end run of the real code on the mock data, without Kafka.

Role in the process (mock data -> unit tests -> implement -> sanity -> next phase): after the
unit tests are green, this module runs every layer built in Phase 2 together, with no stubs and
no mocks of project code:

* simulated fleet as JSONL -> streaming file ingest with checkpoint (``run_ingest_files``) ->
  ``raw_frames`` / ``rejects`` Parquet -> batch transform (``run_transform``) -> ``frames`` /
  ``events`` / ``drives`` Parquet -> DuckDB ``Warehouse`` queries;
* the validator CLI on the Parquet ``raw_frames`` table.

Expected values come from the simulator labels or from an independent in-memory route (the
Phase 1 pandas functions on rows parsed from the JSONL), never from the code under test. The
fleet is built once for the module. Tests run in file order: the two that add data to the same
warehouse (idempotent rerun, new file) come last, and the mutation guard works on a copy.
"""

import json
import shutil
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyspark.sql import SparkSession

from pipeline_helpers import raw_from_frames
from test_sanity_phase1 import FLEET_SEED, N_DRIVES, N_FRAMES, labels_table
from vl.config import DEFAULT
from vl.etl.features import derive_frames, detect_events, summarize_drive
from vl.etl.ingest import run_ingest_files
from vl.etl.transform import run_transform
from vl.mock.drive_sim import simulate_drive, simulate_fleet
from vl.mock.generate import write_fleet_jsonl
from vl.schemas import DRIVES_COLUMNS, EVENTS_COLUMNS, FRAMES_COLUMNS, RAW_ORDER_COLUMNS
from vl.validate import main as validate_main
from vl.warehouse import EXAMPLE_QUERIES, Warehouse

pytestmark = pytest.mark.sanity

ATOL = 1e-9
TABLE_NAMES = ("raw_frames", "rejects", "frames", "events", "drives")
EXPECTED_CODES = {"VL-001", "VL-002", "VL-004", "VL-006"}  # VL-003 never occurs in these runs
NEW_DRIVE_ID = "drive-006"


class Tables(NamedTuple):
    """The three derived tables of one drive."""

    frames: pd.DataFrame
    events: pd.DataFrame
    drives: pd.DataFrame


class Pipeline(NamedTuple):
    """Everything the fleet run produced.

    Attributes:
        spark: The shared Spark session.
        source_dir: Directory the streaming ingest reads (only ``frames.jsonl``).
        fleet_dir: Directory written by ``write_fleet_jsonl`` (``frames.jsonl``, ``labels.json``).
        out_dir: Warehouse directory.
        checkpoint_dir: Streaming checkpoint directory.
        labels: The fleet's fault labels as a table.
        expected: The in-memory (pandas) tables per ``drive_id``.
        warehouse: The DuckDB warehouse over ``out_dir``.
    """

    spark: SparkSession
    source_dir: Path
    fleet_dir: Path
    out_dir: Path
    checkpoint_dir: Path
    labels: pd.DataFrame
    expected: dict[str, Tables]
    warehouse: Warehouse


def pandas_route(raw: pd.DataFrame) -> Tables:
    """Run the Phase 1 functions on the raw rows of one drive, in the documented raw order."""
    ordered = raw.sort_values(RAW_ORDER_COLUMNS, kind="stable", na_position="last")
    frames = derive_frames(ordered.reset_index(drop=True))
    events = detect_events(frames).sort_values(["start_frame", "event_type"])
    return Tables(frames, events.reset_index(drop=True), summarize_drive(frames, events))


def expected_from_jsonl(path: Path) -> dict[str, Tables]:
    """Parse a ``frames.jsonl`` in memory (no Spark) and give the pandas tables per drive."""
    messages = [json.loads(line) for line in path.read_text().splitlines()]
    ids = sorted({message["drive_id"] for message in messages})
    by_drive = {i: [m for m in messages if m["drive_id"] == i] for i in ids}
    return {i: pandas_route(raw_from_frames(frames)) for i, frames in by_drive.items()}


def warehouse_tables(warehouse: Warehouse, drive_id: str) -> Tables:
    """Read the three derived tables of one drive from the warehouse, in key order."""
    orders = {"frames": "frame_idx", "events": "start_frame, event_type", "drives": "drive_id"}
    columns = {"frames": FRAMES_COLUMNS, "events": EVENTS_COLUMNS, "drives": DRIVES_COLUMNS}
    read = [
        f"SELECT {', '.join(columns[t])} FROM {t} WHERE drive_id = ? ORDER BY {orders[t]}"
        for t in Tables._fields
    ]
    return Tables(*(warehouse.query(sql, [drive_id]) for sql in read))


def comparable(table: pd.DataFrame) -> pd.DataFrame:
    """Make a table comparable: JSON ``params`` become numeric columns, string nulls become None."""
    out = table.copy()
    for column in out.select_dtypes(include=object).columns:
        out[column] = out[column].where(out[column].notna(), None)
    if "params" not in out:
        return out
    parsed = pd.json_normalize(out["params"].map(json.loads).tolist())
    return pd.concat([out.drop(columns="params"), parsed], axis=1)


def same_table(got: pd.DataFrame, want: pd.DataFrame) -> bool:
    """Tell whether two tables have the same columns, dtypes and values (floats within ATOL)."""
    try:
        assert_frame_equal(comparable(got), comparable(want), check_exact=False, atol=ATOL, rtol=0)
    except AssertionError:
        return False
    return True


def mismatched_drives(warehouse: Warehouse, expected: dict[str, Tables]) -> list[str]:
    """List the drives whose warehouse tables differ from the in-memory route.

    Args:
        warehouse: The warehouse to check.
        expected: The pandas tables per ``drive_id``.

    Returns:
        The ``drive_id`` values with at least one of ``frames``, ``events``, ``drives`` unequal.
    """
    return [
        drive_id
        for drive_id, want in expected.items()
        if not all(map(same_table, warehouse_tables(warehouse, drive_id), want))
    ]


def row_counts(out_dir: Path) -> dict[str, int]:
    """Count the rows of every table with a fresh warehouse connection."""
    warehouse = Warehouse(str(out_dir))
    return {t: int(warehouse.query(f"SELECT COUNT(*) AS n FROM {t}")["n"][0]) for t in TABLE_NAMES}


def all_frames(warehouse: Warehouse) -> pd.DataFrame:
    """Read the whole ``frames`` table in key order."""
    sql = f"SELECT {', '.join(FRAMES_COLUMNS)} FROM frames ORDER BY drive_id, frame_idx"
    return warehouse.query(sql)


def labels_cte(labels: pd.DataFrame) -> tuple[str, list[object]]:
    """Build a SQL ``WITH l(...)`` clause and its parameters from the label table."""
    cells = ["(CAST(? AS VARCHAR), CAST(? AS BIGINT), CAST(? AS VARCHAR))"] * len(labels)
    params: list[object] = []
    for label in labels.itertuples(index=False):
        code = label.expected_code if isinstance(label.expected_code, str) else None
        params += [label.drive_id, int(label.frame_idx), code]
    return f"WITH l(drive_id, frame_idx, expected) AS (VALUES {', '.join(cells)}) ", params


def ingest_and_transform(
    spark: SparkSession, source_dir: Path, out_dir: Path, checkpoint_dir: Path
) -> None:
    """Run the streaming file ingest and then the batch transform."""
    run_ingest_files(spark, str(source_dir), str(out_dir), str(checkpoint_dir))
    run_transform(spark, str(out_dir), DEFAULT)


def rerun(pipeline: Pipeline) -> None:
    """Ingest and transform again with the pipeline's own source, warehouse and checkpoint."""
    paths = (pipeline.source_dir, pipeline.out_dir, pipeline.checkpoint_dir)
    ingest_and_transform(pipeline.spark, *paths)


@pytest.fixture(scope="module")
def pipeline(spark: SparkSession, tmp_path_factory: pytest.TempPathFactory) -> Pipeline:
    """Simulate the fleet, ingest it from files, transform it and open the warehouse."""
    drives = simulate_fleet(N_DRIVES, seed=FLEET_SEED, n_frames=N_FRAMES)
    root = tmp_path_factory.mktemp("phase2")
    fleet_dir, source_dir = root / "fleet", root / "source"
    write_fleet_jsonl(drives, str(fleet_dir))
    source_dir.mkdir()
    # labels.json must stay out of the source directory: the file source would ingest it.
    shutil.copy(fleet_dir / "frames.jsonl", source_dir / "frames.jsonl")
    labels = labels_table(drives)
    expected = expected_from_jsonl(fleet_dir / "frames.jsonl")
    out_dir, checkpoint_dir = root / "warehouse", root / "checkpoint"
    ingest_and_transform(spark, source_dir, out_dir, checkpoint_dir)
    return Pipeline(
        spark,
        source_dir,
        fleet_dir,
        out_dir,
        checkpoint_dir,
        labels,
        expected,
        Warehouse(str(out_dir)),
    )


def test_ingest_counts_match_the_simulator(pipeline: Pipeline) -> None:
    """Rejects equal the malformed labels; raw_frames rows are the other lines of the file."""
    n_malformed = int((pipeline.labels["kind"] == "malformed").sum())
    n_lines = len((pipeline.fleet_dir / "frames.jsonl").read_text().splitlines())
    counts = row_counts(pipeline.out_dir)
    assert n_malformed > 0
    assert counts["rejects"] == n_malformed
    assert counts["raw_frames"] == n_lines - n_malformed


def test_spark_route_equals_the_pure_pandas_route(pipeline: Pipeline) -> None:
    """For every drive the warehouse tables equal the Phase 1 functions run in memory."""
    assert len(pipeline.expected) == N_DRIVES
    assert mismatched_drives(pipeline.warehouse, pipeline.expected) == []


def test_every_labeled_fault_is_detected_in_the_warehouse(pipeline: Pipeline) -> None:
    """Each coded label gives exactly its code on its frame; unlabeled frames stay <= 1% flagged."""
    coded = pipeline.labels[pipeline.labels["expected_code"].notna()]
    with_labels, params = labels_cte(coded)
    missed = pipeline.warehouse.query(
        with_labels + "SELECT l.drive_id, l.frame_idx, l.expected, f.error_code FROM l "
        "LEFT JOIN frames f ON f.drive_id = l.drive_id AND f.frame_idx = l.frame_idx "
        "WHERE f.error_code IS DISTINCT FROM l.expected",
        params,
    )
    unlabeled = pipeline.warehouse.query(
        with_labels + "SELECT COUNT(*) AS n, COUNT(f.error_code) AS flagged FROM frames f "
        "LEFT JOIN l ON f.drive_id = l.drive_id AND f.frame_idx = l.frame_idx "
        "WHERE l.frame_idx IS NULL",
        params,
    )
    assert "gap_after_malformed" in set(coded["kind"])
    assert missed.empty, missed
    assert unlabeled["n"][0] > N_DRIVES * N_FRAMES * 0.9
    assert unlabeled["flagged"][0] / unlabeled["n"][0] <= DEFAULT.max_false_positive_rate


def test_validator_on_the_parquet_raw_frames_passes(
    pipeline: Pipeline, capsys: pytest.CaptureFixture[str]
) -> None:
    """The validator CLI exits 0 on the Parquet ``raw_frames`` table, with no FAIL line."""
    argv = ["--source", str(pipeline.out_dir / "raw_frames")]
    argv += ["--labels", str(pipeline.fleet_dir / "labels.json")]
    code = validate_main(argv)
    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert not [line for line in lines if line.startswith("FAIL")]


def test_example_queries_run_on_the_fleet_warehouse(pipeline: Pipeline) -> None:
    """Every example query returns rows; the key ones have the shape the spec gives."""
    result = {name: pipeline.warehouse.query(sql) for name, sql in EXAMPLE_QUERIES.items()}
    assert all(not table.empty for table in result.values()), [
        name for name, table in result.items() if table.empty
    ]
    assert len(result["hard_brakes_per_drive"]) == N_DRIVES
    assert set(result["error_counts"]["error_code"]) <= EXPECTED_CODES
    overview = result["drives_overview"]
    assert list(overview["vehicle_id"]) == [f"veh-{i}" for i in overview["drive_id"]]
    assert len(overview) == N_DRIVES
    assert overview["vehicle_kind"].notna().all()
    assert overview["drive_start_utc"].notna().all()


def test_rerun_is_idempotent(pipeline: Pipeline) -> None:
    """A second ingest (checkpoint: no new rows) and transform change no count and no frame."""
    before_counts = row_counts(pipeline.out_dir)
    before_frames = all_frames(pipeline.warehouse)
    rerun(pipeline)
    assert row_counts(pipeline.out_dir) == before_counts
    assert_frame_equal(all_frames(Warehouse(str(pipeline.out_dir))), before_frames)


def test_new_file_is_ingested_once_and_transformed(pipeline: Pipeline, tmp_path: Path) -> None:
    """A new source file adds its drive exactly once and leaves the earlier drives unchanged."""
    drive = simulate_drive(NEW_DRIVE_ID, "scripted", "car", 300, seed=7)
    write_fleet_jsonl([drive], str(tmp_path))
    shutil.copy(tmp_path / "frames.jsonl", pipeline.source_dir / "second.jsonl")
    before = pipeline.warehouse.query("SELECT * FROM drives ORDER BY drive_id")
    rerun(pipeline)
    warehouse = Warehouse(str(pipeline.out_dir))
    after = warehouse.query("SELECT * FROM drives ORDER BY drive_id")
    expected = {**pipeline.expected, **expected_from_jsonl(tmp_path / "frames.jsonl")}
    assert list(after["drive_id"]) == [*before["drive_id"], NEW_DRIVE_ID]
    assert_frame_equal(after.iloc[:-1].reset_index(drop=True), before)
    assert mismatched_drives(warehouse, expected) == []


def test_sanity_detects_a_broken_pipeline(pipeline: Pipeline, tmp_path: Path) -> None:
    """Mutation guard: without one drive's frames partition the equality helper flags that drive."""
    broken = tmp_path / "broken"
    shutil.copytree(pipeline.out_dir, broken)
    victim = sorted(pipeline.expected)[2]
    shutil.rmtree(broken / "frames" / f"drive_id={victim}")
    assert mismatched_drives(Warehouse(str(broken)), pipeline.expected) == [victim]
