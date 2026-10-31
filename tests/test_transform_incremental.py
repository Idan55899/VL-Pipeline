"""Incremental transform: recompute only the drives whose raw data changed, same as a full run."""

import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal
from pyspark.sql import SparkSession

from fixtures.loader import load_golden
from transform_helpers import TABLES, golden_rows, read_pd, write_raw_frames
from vl.config import DEFAULT
from vl.etl import incremental
from vl.etl.incremental import Plan, plan, save_state
from vl.etl.transform import run_transform

A, B, C = "g02_hard_brake", "g05_dropped_frame", "g04_stop"


def _rows(name: str) -> list[tuple[object, ...]]:
    """Raw rows of one golden case (drive id = case name)."""
    return golden_rows(load_golden(name))


def _files(warehouse: Path, drive: str) -> list[str]:
    """Names of the ``frames`` Parquet files of one drive (new names mean rewritten)."""
    return sorted(p.name for p in (warehouse / "frames" / f"drive_id={drive}").glob("*.parquet"))


def _assert_same_tables(spark: SparkSession, got: Path, want: Path) -> None:
    """The three output tables of two warehouses are equal."""
    for table in TABLES:
        assert_frame_equal(read_pd(spark, got, table), read_pd(spark, want, table), obj=table)


def test_incremental_run_equals_a_full_run_and_rewrites_only_changed_drives(
    spark: SparkSession, tmp_path: Path
) -> None:
    inc, full = tmp_path / "inc", tmp_path / "full"
    write_raw_frames(spark, inc, _rows(A) + _rows(B))
    run_transform(spark, str(inc), incremental=True)  # no state yet: a full run
    untouched = _files(inc, A)
    replay = _rows(B)[-3:]  # a replayed batch of drive B, plus a new drive C
    write_raw_frames(spark, inc, replay + _rows(C), mode="append")
    counts = run_transform(spark, str(inc), incremental=True)
    write_raw_frames(spark, full, _rows(A) + _rows(B) + replay + _rows(C))
    assert counts == run_transform(spark, str(full))
    _assert_same_tables(spark, inc, full)
    assert _files(inc, A) == untouched and _files(inc, B) and _files(inc, C)


def test_incremental_run_drops_removed_drives_skips_no_change_and_recomputes_on_new_config(
    spark: SparkSession, tmp_path: Path
) -> None:
    inc, full = tmp_path / "inc", tmp_path / "full"
    write_raw_frames(spark, inc, _rows(A) + _rows(B))
    run_transform(spark, str(inc), incremental=True)
    shutil.rmtree(inc / "raw_frames" / f"drive_id={B}")
    run_transform(spark, str(inc), incremental=True)
    write_raw_frames(spark, full, _rows(A))
    run_transform(spark, str(full))
    _assert_same_tables(spark, inc, full)
    before = _files(inc, A)
    run_transform(spark, str(inc), incremental=True)  # nothing changed: nothing rewritten
    assert _files(inc, A) == before
    run_transform(spark, str(inc), replace(DEFAULT, hard_brake_ms2=-6.0), incremental=True)
    assert _files(inc, A) != before  # new thresholds: every drive recomputed
    assert read_pd(spark, inc, "events").empty  # g02's brake is gentler than -6 m/s^2


def test_a_change_to_the_pipeline_code_forces_a_full_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same raw files and thresholds, different code: the old outputs are not trusted."""
    assert all(path.is_file() for path in incremental.CODE_FILES)  # the real list is not stale
    code = tmp_path / "features.py"
    code.write_text("THRESHOLD = 1\n")
    monkeypatch.setattr(incremental, "CODE_FILES", (code,))
    for table in incremental.OUTPUTS:
        (tmp_path / table).mkdir()
    save_state(tmp_path, DEFAULT)
    assert plan(tmp_path, DEFAULT) == Plan([], [])  # nothing changed: nothing to do
    code.write_text("THRESHOLD = 2\n")
    assert plan(tmp_path, DEFAULT) is None


def _old_table_delete_spy(
    spark: SparkSession, warehouse: Path, seen: list[set[object]]
) -> Callable[..., None]:
    """Build a ``shutil.rmtree`` replacement that records who is live before an ``.old`` delete.

    Args:
        spark: The shared Spark session.
        warehouse: Warehouse directory.
        seen: Appended with the live `drive_id` set of the matching table on each such delete.

    Returns:
        A drop-in replacement for ``shutil.rmtree`` that still deletes.
    """
    real_rmtree = shutil.rmtree

    def spy(path: object, *args: object, **kwargs: object) -> None:
        name = Path(str(path)).name
        if name in ("events.old", "drives.old"):
            live = read_pd(spark, warehouse, name.removesuffix(".old"))
            seen.append(set(live["drive_id"]))
        real_rmtree(path, *args, **kwargs)

    return spy


def test_merge_flat_never_deletes_the_live_table_before_the_new_one_is_in_place(
    spark: SparkSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old events/drives bytes are discarded only after the new ones are already live.

    Guards the swap in ``_merge_flat``: an earlier version called ``shutil.rmtree`` on the live
    ``events``/``drives`` directory before renaming the new one in, so the table did not exist on
    disk for however long the delete took. The fix renames the live directory out of the way
    first (an atomic, near-instant metadata operation), renames the new one into its place, and
    only then deletes the old one, now safely out of any reader's path.
    """
    warehouse = tmp_path / "wh"
    write_raw_frames(spark, warehouse, _rows(A) + _rows(B))
    run_transform(spark, str(warehouse), incremental=True)
    write_raw_frames(spark, warehouse, _rows(C), mode="append")  # a new drive: triggers a merge

    seen_live_drives: list[set[object]] = []
    monkeypatch.setattr(shutil, "rmtree", _old_table_delete_spy(spark, warehouse, seen_live_drives))
    run_transform(spark, str(warehouse), incremental=True)

    # by the time the old bytes were irreversibly discarded, the live path already served C
    assert seen_live_drives and all(C in drives for drives in seen_live_drives)
