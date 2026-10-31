"""Hardening tests for the Spark ingest: line-boundary bypasses, settings, write/parse parity.

These complement ``test_ingest.py`` (the approved contract). The bypass cases come from an
independent review: ``from_json`` stops reading at the first closing bracket, so text after
a ``]`` must never let a line through as a valid frame.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from fixtures.message_loader import malformed_lines, valid_lines
from ingest_helpers import frame_key, message, raw_df, table_rows
from vl.etl import ingest
from vl.etl.ingest import parse_messages, write_batch
from vl.spark import get_spark

VALID = message("drive-A", 0, [[1, 2, 3]])
OTHER = message("drive-B", 1, [[4, 5, 6]])
BYPASS_LINES = {
    "one_bracket": VALID + "]",
    "three_brackets": VALID + "]]]",
    "bracket_then_text": VALID + "] garbage text",
    "bracket_then_broken": VALID + "], {broken",
    "bracket_then_sql": VALID + "];DROP TABLE x",
    "bracket_then_second_message": VALID + "] [" + OTHER,
    "trailing_null_then_bracket": VALID + ",null]",
    "trailing_nulls_then_brackets": VALID + ",null]]",
    "two_messages_comma": VALID + "," + OTHER,
    "wrapped_in_array": "[" + VALID + "]",
}


def test_text_after_a_closing_bracket_is_rejected(spark: SparkSession) -> None:
    valid, rejects = parse_messages(raw_df(spark, list(BYPASS_LINES.values())))
    assert valid.count() == 0, "a bypass line was accepted as a valid frame"
    got = Counter((r["payload"], r["error_code"]) for r in rejects.collect())
    for name, line in BYPASS_LINES.items():
        assert got[(line, "VL-005")] == 1, f"{name}: {line!r} not rejected verbatim as VL-005"
    assert sum(got.values()) == len(BYPASS_LINES)


def test_clean_and_padded_lines_stay_valid(spark: SparkSession) -> None:
    padded = {"clean": VALID, "padded": "  " + VALID + " \t", "newline": VALID + "\n"}
    valid, rejects = parse_messages(raw_df(spark, list(padded.values())))
    assert [r["payload"] for r in rejects.collect()] == [], "a clean or padded line was rejected"
    assert [r["drive_id"] for r in valid.collect()] == ["drive-A"] * len(padded), list(padded)


def test_write_batch_writes_what_parse_messages_returns(
    spark: SparkSession, tmp_path: Path
) -> None:
    lines = valid_lines() + malformed_lines() + list(BYPASS_LINES.values())
    valid, rejects = parse_messages(raw_df(spark, lines))
    write_batch(raw_df(spark, lines), 0, str(tmp_path))
    expected_valid = sorted((tuple(r) for r in valid.collect()), key=frame_key)
    assert table_rows(spark, tmp_path, "raw_frames") == expected_valid
    written = table_rows(spark, tmp_path, "rejects")
    assert sorted((r[0], r[1]) for r in written) == sorted(
        (r["payload"], r["error_code"]) for r in rejects.collect()
    )


def test_get_spark_on_a_live_session(spark: SparkSession) -> None:
    with pytest.raises(RuntimeError, match="Kafka"):
        get_spark("needs-kafka", with_kafka=True)
    again = get_spark("plain")
    assert again.sparkContext.applicationId == spark.sparkContext.applicationId


def _crash_before_rejects(publish: Callable[[Path, Path, str], None]) -> Callable[..., None]:
    """Wrap ``_publish`` so a batch crashes after ``raw_frames`` is in place, before ``rejects``.

    Args:
        publish: The real ``_publish``.

    Returns:
        A replacement that raises ``RuntimeError`` for the ``rejects`` table.
    """

    def crash(staged: Path, table: Path, tag: str) -> None:
        if table.name == "rejects":
            raise RuntimeError("executor lost")
        publish(staged, table, tag)

    return crash


def test_a_replayed_batch_replaces_its_own_rows_instead_of_adding_them(
    spark: SparkSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spark replays a micro-batch whose checkpoint commit did not happen, with the same id.

    Guards against the duplicate rows a replay used to append: every replayed frame was then
    flagged VL-004 and left out of the kinematics, which could hide a drive's events.
    """
    lines = [VALID, OTHER, "not json"]
    real_publish = ingest._publish
    monkeypatch.setattr(ingest, "_publish", _crash_before_rejects(real_publish))
    with pytest.raises(RuntimeError):
        write_batch(raw_df(spark, lines), 0, str(tmp_path), "run-a")
    monkeypatch.setattr(ingest, "_publish", real_publish)
    for _ in range(2):  # the replay after the crash, then a replay of a completed batch
        write_batch(raw_df(spark, lines), 0, str(tmp_path), "run-a")
        assert len(table_rows(spark, tmp_path, "raw_frames")) == 2
        assert len(table_rows(spark, tmp_path, "rejects")) == 1
    write_batch(raw_df(spark, lines), 0, str(tmp_path), "run-b")  # a new checkpoint restarts at 0
    assert len(table_rows(spark, tmp_path, "raw_frames")) == 4
