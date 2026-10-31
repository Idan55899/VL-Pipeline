"""Tests for the Spark ingest (``vl.etl.ingest``) against the approved message fixtures.

Expected values come from ``tests/fixtures/messages/expected.json`` (hand-derived) and
from hand-written messages below; nothing is derived by running code under ``src/vl``.
"""

import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from fixtures.message_loader import (
    ValidCase,
    load_malformed_cases,
    load_valid_cases,
    valid_lines,
)
from ingest_helpers import (
    assert_row_matches,
    assert_rows_match,
    case_row,
    ddl_dtypes,
    frame_key,
    message,
    parquet_files,
    raw_df,
    read_table,
    table_rows,
    write_lines,
)
from vl.etl.ingest import parse_messages, run_ingest_files, write_batch
from vl.schemas import RAW_FRAMES_DDL, REJECTS_DDL

VALID = load_valid_cases()
MALFORMED = load_malformed_cases()
# Hand-written malformed lines beyond the fixtures (payloads must be preserved verbatim).
EXTRA_REJECTS = {
    "padded_not_json": "  not json  ",
    "blank_spaces": "   ",
    "ts_null": '{"drive_id":"d","frame_idx":0,"points":[[1,2,3]],"ts":null}',
    "ts_true": '{"drive_id":"d","frame_idx":0,"points":[[1,2,3]],"ts":true}',
    "null_point": '{"drive_id":"d","frame_idx":0,"points":[[1,2,3],null],"ts":0.0}',
    "frame_idx_beyond_int64": (
        '{"drive_id":"d","frame_idx":9223372036854775808,"points":[[1,2,3]],"ts":0.0}'
    ),
}
# The file source hands EVERY line to parse_messages, so a blank line is a reject too.
ALL_REJECTS = [c.text for c in MALFORMED.values()] + list(EXTRA_REJECTS.values())
BAD_METADATA: dict[str, dict[str, object]] = {
    "vehicle_id_number": {"vehicle_id": 5},
    "vehicle_kind_bool": {"vehicle_kind": True},
    "drive_start_utc_number": {"drive_start_utc": 123},
}


def test_valid_fixture_cases_parse_to_expected(spark: SparkSession) -> None:
    """Every valid fixture case, parsed in one Spark call, gives its hand-derived row."""
    valid, rejects = parse_messages(raw_df(spark, valid_lines()))
    rows = [tuple(r) for r in valid.collect()]
    assert rejects.count() == 0
    assert len(rows) == len(VALID)
    assert valid.dtypes == ddl_dtypes(spark, RAW_FRAMES_DDL)
    by_key = {frame_key(row): row for row in rows}
    assert len(by_key) == len(VALID)
    for name, case in VALID.items():
        expected = case_row(case)
        assert frame_key(expected) in by_key, f"{name}: no row parsed"
        assert_row_matches(by_key[frame_key(expected)], expected, name)
    # Hand-written expectations, independent of expected.json: the empty cloud has no
    # centroid, and integer coordinates beyond int32 centimeters do not overflow.
    assert by_key[frame_key(case_row(VALID["v08_empty_cloud"]))][6:] == (0, None, None, None)
    big = by_key[frame_key(case_row(VALID["v09_large_integer_coordinate"]))]
    assert big[6:] == (2, 20000000.0, 0.0, 0.0)


def _assert_labelled_payloads_kept(payloads: Counter[str]) -> None:
    """Assert every malformed fixture and hand-written line is a reject payload, verbatim."""
    labelled = {f"{n}: {c.reason}": c.text for n, c in MALFORMED.items()} | EXTRA_REJECTS
    for name, text in labelled.items():
        assert payloads[text] >= 1, f"{name}: payload {text!r} not kept verbatim"


def test_malformed_lines_go_to_rejects_verbatim(spark: SparkSession) -> None:
    """Fixture and hand-written malformed lines, mixed with valid ones, all become VL-005."""
    lines = valid_lines() + ALL_REJECTS
    before = datetime.now() - timedelta(minutes=1)
    valid, rejects = parse_messages(raw_df(spark, lines))
    rows = rejects.collect()
    assert valid.count() == len(VALID), "the valid lines must all be kept"
    assert len(rows) == len(ALL_REJECTS)
    assert rejects.dtypes == ddl_dtypes(spark, REJECTS_DDL)
    assert sorted(r["payload"] for r in rows) == sorted(ALL_REJECTS)
    assert {r["error_code"] for r in rows} == {"VL-005"}
    _assert_labelled_payloads_kept(Counter(r["payload"] for r in rows))
    for r in rows:
        assert before <= r["ingested_at"] <= datetime.now() + timedelta(minutes=1), r["payload"]


def test_metadata_type_strictness_and_null_is_absent(spark: SparkSession) -> None:
    bad = {
        name: message("drive-A", 0, [[1, 2, 3]], **extra) for name, extra in BAD_METADATA.items()
    }
    null_line = message("drive-A", 0, [[1, 2, 3]], vehicle_id=None, vehicle_kind=None)
    assert '"vehicle_id": null' in null_line
    valid, rejects = parse_messages(raw_df(spark, [*bad.values(), null_line]))
    assert [tuple(r)[:4] for r in valid.collect()] == [("drive-A", None, None, None)]
    rejected = sorted(r["payload"] for r in rejects.collect())
    assert rejected == sorted(bad.values()), f"non-string metadata cases: {list(bad)}"


def test_empty_input(spark: SparkSession) -> None:
    valid, rejects = parse_messages(raw_df(spark, []))
    assert valid.count() == 0
    assert rejects.count() == 0
    assert valid.dtypes == ddl_dtypes(spark, RAW_FRAMES_DDL)
    assert rejects.dtypes == ddl_dtypes(spark, REJECTS_DDL)


def _two_drive_batch() -> list[str]:
    """Messages of drives ``drive-A``, ``007`` (numeric-looking) plus one malformed line."""
    return [
        message("drive-A", 0, [[0, 0, 0], [100, 200, 300]]),
        message("drive-A", 1, [[100, 0, 0]], vehicle_id="veh-1", vehicle_kind="car"),
        message("007", 0, [[-50, 50, 25]]),
        "not json",
    ]


def test_write_batch_appends_partitioned_by_drive(spark: SparkSession, tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_batch(raw_df(spark, _two_drive_batch()), 0, str(out))
    write_batch(raw_df(spark, [message("drive-A", 2, [[300, 0, 0]])]), 1, str(out))
    assert {p.name for p in (out / "raw_frames").glob("drive_id=*")} == {
        "drive_id=drive-A",
        "drive_id=007",
    }
    rows = table_rows(spark, out, "raw_frames")
    assert [(r[0], r[4], r[6]) for r in rows] == [
        ("007", 0, 1),
        ("drive-A", 0, 2),
        ("drive-A", 1, 1),
        ("drive-A", 2, 1),
    ]
    assert_row_matches(rows[1], ("drive-A", None, None, None, 0, 0.0, 2, 0.5, 1.0, 1.5))
    assert_row_matches(rows[0], ("007", None, None, None, 0, 0.0, 1, -0.5, 0.5, 0.25))
    assert_row_matches(rows[2], ("drive-A", "veh-1", "car", None, 1, 0.0625, 1, 1.0, 0.0, 0.0))
    assert len(table_rows(spark, out, "rejects")) == 1
    back = read_table(spark, out, "raw_frames")
    assert back is not None
    assert dict(back.dtypes)["drive_id"] == "string"  # "007" is not inferred as a number
    assert {r["drive_id"] for r in back.collect()} == {"007", "drive-A"}


def test_write_batch_empty_batch_writes_nothing_and_keeps_earlier_output(
    spark: SparkSession, tmp_path: Path
) -> None:
    write_batch(raw_df(spark, []), 0, str(tmp_path / "fresh"))
    assert list((tmp_path / "fresh").rglob("*.parquet")) == []
    out = tmp_path / "kept"
    write_batch(raw_df(spark, _two_drive_batch()), 0, str(out))
    write_batch(raw_df(spark, []), 1, str(out))
    assert len(table_rows(spark, out, "raw_frames")) == 3
    assert len(table_rows(spark, out, "rejects")) == 1


class FileRun:
    """A file-source ingest workspace: source, output and checkpoint directories."""

    def __init__(self, spark: SparkSession, root: Path) -> None:
        self.spark = spark
        self.source = root / "source"
        self.out = root / "out"
        self.checkpoint = root / "checkpoint"

    def add(self, name: str, lines: list[str]) -> None:
        """Drop a message file into the source directory."""
        write_lines(self.source, name, lines)

    def run(self) -> None:
        """Run the file ingest once over the source directory."""
        run_ingest_files(self.spark, str(self.source), str(self.out), str(self.checkpoint))

    def raw(self) -> list[tuple[object, ...]]:
        """Read back ``raw_frames`` as sorted tuples."""
        return table_rows(self.spark, self.out, "raw_frames")

    def rejects(self) -> list[tuple[object, ...]]:
        """Read back ``rejects`` as sorted tuples."""
        return table_rows(self.spark, self.out, "rejects")


@pytest.fixture
def file_run(spark: SparkSession, tmp_path: Path) -> FileRun:
    return FileRun(spark, tmp_path)


def _first_half() -> list[ValidCase]:
    return list(VALID.values())[:5]


def _second_half() -> list[ValidCase]:
    return list(VALID.values())[5:]


def test_ingest_files_each_file_once_and_new_files_picked_up(file_run: FileRun) -> None:
    file_run.add("a.jsonl", [c.text for c in _first_half()])
    file_run.run()
    assert_rows_match(file_run.raw(), _first_half())
    file_run.add("b.jsonl", [c.text for c in _second_half()] + ALL_REJECTS)
    file_run.run()
    assert_rows_match(file_run.raw(), list(VALID.values()))
    assert [r[0] for r in file_run.rejects()] == sorted(ALL_REJECTS)
    before = (file_run.raw(), file_run.rejects(), parquet_files(file_run.out))
    file_run.run()  # nothing new: no duplicate rows, no new Parquet file
    assert (file_run.raw(), file_run.rejects(), parquet_files(file_run.out)) == before


def test_ingest_empty_directory(file_run: FileRun) -> None:
    file_run.source.mkdir(parents=True)
    file_run.run()
    assert file_run.raw() == []
    assert file_run.rejects() == []


def test_late_arrival_order_does_not_change_results(spark: SparkSession, tmp_path: Path) -> None:
    lines = valid_lines() + ALL_REJECTS
    shuffled = lines[:]
    random.Random(7).shuffle(shuffled)
    assert shuffled != lines
    ordered_run = FileRun(spark, tmp_path / "ordered")
    shuffled_run = FileRun(spark, tmp_path / "shuffled")
    ordered_run.add("m.jsonl", lines)
    shuffled_run.add("m.jsonl", shuffled)
    ordered_run.run()
    shuffled_run.run()
    assert shuffled_run.raw() == ordered_run.raw()
    assert [r[0] for r in shuffled_run.rejects()] == [r[0] for r in ordered_run.rejects()]
    assert_rows_match(shuffled_run.raw(), list(VALID.values()))


def test_late_arrival_across_files_does_not_change_results(
    spark: SparkSession, tmp_path: Path
) -> None:
    lines = valid_lines() + ALL_REJECTS
    first, second = lines[: len(lines) // 2], lines[len(lines) // 2 :]
    in_order = FileRun(spark, tmp_path / "in_order")
    reversed_ = FileRun(spark, tmp_path / "reversed")
    for run, (name_1, part_1, name_2, part_2) in (
        (in_order, ("a.jsonl", first, "b.jsonl", second)),
        (reversed_, ("b.jsonl", second, "a.jsonl", first)),
    ):
        run.add(name_1, part_1)
        run.run()
        run.add(name_2, part_2)
        run.run()
    assert reversed_.raw() == in_order.raw()
    assert [r[0] for r in reversed_.rejects()] == [r[0] for r in in_order.rejects()]
    assert_rows_match(reversed_.raw(), list(VALID.values()))
