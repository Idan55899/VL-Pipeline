"""Hardening tests for the transform, the warehouse and the Parquet validator source.

They pin what the first review round found: determinism of conflicting duplicate frames,
independence from the ambient partition overwrite mode, special characters in warehouse paths,
a clear error for a Parquet file without a DDL column, and the command line edge cases.
"""

import os
import random
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyspark.sql import SparkSession

from fixtures.loader import load_golden
from pipeline_helpers import frames_table, raw_from_frames
from transform_helpers import RawRow, golden_rows, read_pd, transform_rows, write_raw_frames
from vl.config import DEFAULT
from vl.etl.transform import main, run_transform
from vl.mock.drive_sim import SimConfig, simulate_drive
from vl.validate import _derive_fleet, _read_parquet_source
from vl.warehouse import Warehouse

G01 = "g01_constant_speed"
G02 = "g02_hard_brake"
G06 = "g06_position_jump"
OVERWRITE_MODE = "spark.sql.sources.partitionOverwriteMode"
SHUFFLES = range(5)
REPO = Path(__file__).resolve().parent.parent


# --- conflicting duplicates are resolved deterministically ---------------------------------


def with_conflicting_copies(rows: list[RawRow]) -> list[RawRow]:
    """Add a copy of frame 10 with another x, and two copies of frame 20 differing in metadata."""
    moved = rows[10][:7] + (rows[10][7] + 0.5,) + rows[10][8:]
    first = rows[20][:1] + ("veh-A",) + rows[20][2:]
    second = rows[20][:1] + ("veh-B",) + rows[20][2:]
    return [*rows, moved, first, second]


def shuffled_batches(rows: list[RawRow], seed: int) -> list[list[RawRow]]:
    """Shuffle the rows and cut them into 2 to 4 batches (the arrival order of an ingest)."""
    rng = random.Random(seed)
    mixed = rows[:]
    rng.shuffle(mixed)
    cuts = sorted(rng.sample(range(1, len(mixed)), rng.randint(1, 3)))
    return [mixed[a:b] for a, b in zip([0, *cuts], [*cuts, len(mixed)])]


def test_conflicting_duplicates_give_identical_frames_for_every_arrival_order(
    spark: SparkSession, tmp_path: Path
) -> None:
    rows = with_conflicting_copies(golden_rows(load_golden(G01)))
    outputs = []
    for seed in SHUFFLES:
        warehouse = tmp_path / f"seed{seed}"
        for number, batch in enumerate(shuffled_batches(rows, seed)):
            write_raw_frames(spark, warehouse, batch, "append" if number else "overwrite")
        run_transform(spark, str(warehouse), DEFAULT)
        outputs.append(read_pd(spark, warehouse, "frames"))
    for other in outputs[1:]:
        assert_frame_equal(other, outputs[0])
    kept = outputs[0].set_index("frame_idx")
    assert kept.loc[10, "x"] == pytest.approx(rows[10][7])  # the lowest x of the tied copies
    assert kept.loc[20, "vehicle_id"] == "veh-A"  # metadata ties: lowest value, null last
    assert len(outputs[0]) == 60


def conflicting_raw_table() -> pd.DataFrame:
    """Raw rows of one simulated drive plus conflicting copies of two frames."""
    raw = raw_from_frames(simulate_drive("dup", n_frames=120, sim=SimConfig.clean()).frames)
    moved = raw.iloc[[30]].assign(x=raw.iloc[30]["x"] + 25.0)
    renamed = raw.iloc[[40]].assign(vehicle_id="other-vehicle")
    return pd.concat([raw, moved, renamed], ignore_index=True)


def write_parquet_batches(raw: pd.DataFrame, table: Path, seed: int) -> None:
    """Write shuffled rows as several Parquet files of a hive-partitioned ``raw_frames`` table."""
    mixed = raw.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    for batch in (mixed.iloc[: len(mixed) // 3], mixed.iloc[len(mixed) // 3 :]):
        batch.to_parquet(table, partition_cols=["drive_id"])


def test_parquet_source_duplicates_give_identical_frames_for_every_file_order(
    tmp_path: Path,
) -> None:
    raw = conflicting_raw_table()
    derived = []
    for seed in SHUFFLES:
        table = tmp_path / f"seed{seed}" / "raw_frames"
        write_parquet_batches(raw, table, seed)
        source = _read_parquet_source(table)
        derived.append(_derive_fleet(source.raw, DEFAULT))
    for other in derived[1:]:
        assert_frame_equal(other, derived[0])
    kept = derived[0].set_index("frame_idx")
    assert kept.loc[30, "x"] == pytest.approx(raw.iloc[30]["x"])  # lowest x of the tied copies
    assert kept.loc[40, "vehicle_id"] == min(raw.iloc[40]["vehicle_id"], "other-vehicle")


# --- static overwrite does not depend on the session configuration -------------------------


@pytest.fixture
def dynamic_overwrite(spark: SparkSession) -> Iterator[None]:
    """Switch the shared session to dynamic partition overwrite and restore it afterwards."""
    previous = spark.conf.get(OVERWRITE_MODE)
    spark.conf.set(OVERWRITE_MODE, "dynamic")
    try:
        yield
    finally:
        spark.conf.set(OVERWRITE_MODE, previous)


@pytest.mark.usefixtures("dynamic_overwrite")
def test_stale_partitions_are_removed_even_with_dynamic_overwrite(
    spark: SparkSession, tmp_path: Path
) -> None:
    rows = golden_rows(load_golden(G02)) + golden_rows(load_golden(G06))
    transform_rows(spark, tmp_path, rows)
    shutil.rmtree(tmp_path / "raw_frames" / f"drive_id={G06}")  # the drive vanishes
    counts = run_transform(spark, str(tmp_path), DEFAULT)
    assert sorted(p.name for p in (tmp_path / "frames").glob("drive_id=*")) == [f"drive_id={G02}"]
    assert counts == {"frames": 160, "events": 1, "drives": 1}
    assert set(read_pd(spark, tmp_path, "frames")["drive_id"]) == {G02}


# --- warehouse paths and files ---------------------------------------------------------------


def write_frames(table_dir: Path, ids: list[str], **overrides: str) -> None:
    """Write a hive-partitioned ``frames`` table with 3 hand-made rows per drive id."""
    table = pd.concat([frames_table(3, drive_id=i, **overrides) for i in ids])
    table.to_parquet(table_dir / "frames", partition_cols=["drive_id"])


def test_warehouse_path_with_special_characters(tmp_path: Path) -> None:
    for name in ("br[x]", "with space", "it's", "star*?[!a]"):
        directory = tmp_path / name
        directory.mkdir()
        write_frames(directory, ["007", "12"])
        warehouse = Warehouse(str(directory))
        found = warehouse.query(
            "select distinct drive_id, typeof(drive_id) t from frames order by 1"
        )
        assert found["drive_id"].tolist() == ["007", "12"], name
        assert set(found["t"]) == {"VARCHAR"}, name
        assert warehouse.query("select count(*) c from events")["c"][0] == 0, name  # typed view


def test_parquet_file_without_a_ddl_column_gives_a_clear_error(tmp_path: Path) -> None:
    table = frames_table(3).drop(columns=["speed"])
    table.to_parquet(tmp_path / "frames", partition_cols=["drive_id"])
    with pytest.raises(ValueError, match=r"frames.*speed") as raised:
        Warehouse(str(tmp_path))
    assert "\n" not in str(raised.value)


# --- the command line ------------------------------------------------------------------------


def test_main_rejects_a_warehouse_path_that_is_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "warehouse.txt"
    path.write_text("not a directory")
    assert main(["--warehouse", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert len(captured.err.strip().splitlines()) == 1


def test_main_rejects_an_invalid_env_value_before_starting_spark(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VL_MAX_JUMP_M", "abc")
    assert main(["--warehouse", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ") and len(captured.err.strip().splitlines()) == 1


def test_main_notes_a_missing_or_empty_raw_frames(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    for kind in ("missing", "empty"):
        warehouse = tmp_path / kind
        warehouse.mkdir()
        if kind == "empty":
            (warehouse / "raw_frames").mkdir()
        done = subprocess.run(
            [sys.executable, "-m", "vl.etl.transform", "--warehouse", str(warehouse)],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO,
            timeout=240,
        )
        assert done.returncode == 0, f"{kind}: {done.stderr}"
        notices = [line for line in done.stderr.splitlines() if line.startswith("notice: ")]
        assert len(notices) == 1, f"{kind}: {done.stderr}"
        assert "raw_frames" in notices[0], kind
        assert done.stdout.strip() == "frames: 0, events: 0, drives: 0", kind
