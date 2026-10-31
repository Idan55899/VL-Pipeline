"""Batch transform: ``raw_frames`` Parquet to the ``frames``, ``events`` and ``drives`` tables.

Role in the pipeline: the second Spark stage. It derives the per-frame kinematics and error
codes of every drive with ``derive_frames`` (inside ``applyInPandas``), detects the driving
events and summarizes each drive, so the warehouse holds the tables the query layer reads.

Design notes:

* Full recompute by default. A run reads all of ``raw_frames`` and overwrites the three output
  tables completely (``mode("overwrite")`` with Spark's default static partition overwrite,
  never ``append`` and never dynamic partition overwrite). A rerun is therefore idempotent, a
  drive that vanished from ``raw_frames`` loses its ``frames`` partition directory, and an
  empty or missing ``raw_frames`` clears the old outputs.
* Incremental mode (``--incremental``). Only drives whose ``raw_frames`` files changed since
  the last run are recomputed, and drives that left ``raw_frames`` are removed (bookkeeping in
  `vl.etl.incremental`). Their ``frames`` partitions are deleted and rewritten; the small
  ``events`` and ``drives`` tables keep the rows of the other drives and get the new ones
  (written to ``<table>.next``, then swapped in). The result equals a full run. A full run
  happens instead when there is no state yet, the thresholds changed or an output is missing.
* ``drive_id`` is a string everywhere: the schemas are passed explicitly (the partition
  column is typed by the DDL, not inferred), so ids such as ``"007"`` keep their zeros.
* ``events`` and ``drives`` are derived from the ``frames`` table just written (read back with
  ``FRAMES_DDL``), so ``derive_frames`` runs once per drive and the stored table is the single
  truth the other two are computed from.
* A missing ``raw_frames`` directory is read as an empty table with ``RAW_FRAMES_DDL``, so
  the empty case flows through the same code and writes typed, zero-row ``events`` and
  ``drives`` files.

Usage: ``python -m vl.etl.transform --warehouse data/warehouse [--incremental]``
"""

import argparse
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from vl.config import DEFAULT, Config, load_config
from vl.etl.features import derive_frames, detect_events, summarize_drive
from vl.etl.incremental import Plan, drive_id_of, plan, save_state
from vl.schemas import (
    DRIVES_DDL,
    EVENTS_DDL,
    FRAMES_DDL,
    RAW_FRAMES_DDL,
    RAW_ORDER_COLUMNS,
    TABLE_DDLS,
)
from vl.spark import get_spark

KEY = "drive_id"
OUTPUTS = ("frames", "events", "drives")
# Writer option that pins static overwrite: the whole table directory is replaced, whatever the
# session's ``spark.sql.sources.partitionOverwriteMode`` says.
OVERWRITE_MODE = "partitionOverwriteMode"


def _frames_of(cfg: Config) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build the per-drive function that derives the ``frames`` rows.

    Args:
        cfg: Pipeline thresholds.

    Returns:
        A function from the raw rows of one drive to its ``frames`` rows; the rows are first
        sorted by ``RAW_ORDER_COLUMNS`` (see `run_transform`).
    """

    def derive(raw: pd.DataFrame) -> pd.DataFrame:
        """Derive the frames of one drive (runs in a Spark Python worker)."""
        ordered = raw.sort_values(RAW_ORDER_COLUMNS, kind="stable", na_position="last")
        return derive_frames(ordered.reset_index(drop=True), cfg)

    return derive


def _events_of(cfg: Config) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build the per-drive function that detects the ``events`` rows.

    Args:
        cfg: Pipeline thresholds.

    Returns:
        A function from the ``frames`` rows of one drive to its ``events`` rows.
    """

    def detect(frames: pd.DataFrame) -> pd.DataFrame:
        """Detect the events of one drive (runs in a Spark Python worker)."""
        return detect_events(frames.sort_values("frame_idx", ignore_index=True), cfg)

    return detect


def _drive_of(cfg: Config) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build the per-drive function that summarizes a drive into its ``drives`` row.

    The events are detected again inside the group (a cheap pass over one drive) because
    ``summarize_drive`` needs them for ``n_events``; this avoids a join between the two tables.

    Args:
        cfg: Pipeline thresholds.

    Returns:
        A function from the ``frames`` rows of one drive to its ``drives`` row.
    """

    def summarize(frames: pd.DataFrame) -> pd.DataFrame:
        """Summarize one drive (runs in a Spark Python worker)."""
        ordered = frames.sort_values("frame_idx", ignore_index=True)
        return summarize_drive(ordered, detect_events(ordered, cfg))

    return summarize


def _read_raw(spark: SparkSession, warehouse_dir: str) -> DataFrame:
    """Read the ``raw_frames`` table, or an empty one when the directory does not exist.

    The schema is explicit, so ``drive_id`` (the partition column) is a string and an empty
    directory or table reads as zero rows instead of failing schema inference.

    Args:
        spark: The Spark session.
        warehouse_dir: Warehouse directory (a local path).

    Returns:
        The raw rows with ``RAW_FRAMES_DDL``.
    """
    path = Path(warehouse_dir) / "raw_frames"
    if not path.is_dir():
        return spark.createDataFrame([], RAW_FRAMES_DDL)
    return spark.read.schema(RAW_FRAMES_DDL).parquet(str(path))


def _read_table(spark: SparkSession, warehouse_dir: str, table: str) -> DataFrame:
    """Read an output table back with its DDL (also when it has no data files).

    Args:
        spark: The Spark session.
        warehouse_dir: Warehouse directory.
        table: ``frames``, ``events`` or ``drives``.

    Returns:
        The table with the columns and types of its DDL.
    """
    return spark.read.schema(TABLE_DDLS[table]).parquet(str(Path(warehouse_dir) / table))


def _write_frames(frames: DataFrame, warehouse_dir: str) -> None:
    """Overwrite the ``frames`` table, partitioned by ``drive_id``.

    Static partition overwrite replaces the whole directory, so partitions of drives that are
    gone disappear. Spark writes no file for an empty partitioned frame (it cannot infer a
    schema), so the directory is created explicitly: readers with an explicit schema then see
    an empty table instead of a missing path.

    Args:
        frames: The derived frames of all drives.
        warehouse_dir: Warehouse directory.
    """
    path = Path(warehouse_dir) / "frames"
    writer = frames.write.mode("overwrite").option(OVERWRITE_MODE, "static")
    writer.partitionBy(KEY).parquet(str(path))
    path.mkdir(parents=True, exist_ok=True)


def _write_flat(table: DataFrame, warehouse_dir: str, name: str) -> None:
    """Overwrite an unpartitioned output table.

    Args:
        table: The rows to store.
        warehouse_dir: Warehouse directory.
        name: ``events`` or ``drives``.
    """
    writer = table.write.mode("overwrite").option(OVERWRITE_MODE, "static")
    writer.parquet(str(Path(warehouse_dir) / name))


def run_transform(
    spark: SparkSession, warehouse_dir: str, cfg: Config = DEFAULT, incremental: bool = False
) -> dict[str, int]:
    """Recompute the ``frames``, ``events`` and ``drives`` tables from ``raw_frames``.

    Reads ``<warehouse_dir>/raw_frames`` and writes ``frames`` (partitioned by ``drive_id``),
    ``events`` and ``drives`` (unpartitioned) with ``mode("overwrite")``, so a rerun recomputes
    the tables and never appends. ``frames`` comes from ``groupBy("drive_id").applyInPandas``
    calling ``derive_frames`` with ``cfg``; ``events`` and ``drives`` are computed per drive from
    ``frames`` with ``detect_events`` and ``summarize_drive``. A missing or empty ``raw_frames``
    gives three zero-row tables; the output dtypes equal ``FRAMES_DDL``, ``EVENTS_DDL`` and
    ``DRIVES_DDL``.

    By default the run is a full recompute: the overwrite replaces each table directory
    entirely, so the ``frames`` partitions of drives that vanished from ``raw_frames`` are removed
    and an empty or missing ``raw_frames`` clears the old outputs. ``drive_id`` is a string
    throughout. The writers pin ``partitionOverwriteMode=static``, so this does not depend on the
    session configuration.

    ``raw_frames`` has no emission-order column and Spark delivers the rows in arbitrary order,
    so each drive's rows are sorted by ``RAW_ORDER_COLUMNS`` (``frame_idx``, ``ts``,
    ``n_points``, ``x``, ``y``, ``z`` and the metadata, nulls last) before ``derive_frames``.
    Conflicting copies of one ``frame_idx`` are therefore resolved the same way on every run:
    the kept row is the one with the lowest ``ts`` (then the lowest following column), not the
    first emitted. Exact replays are unaffected.

    The three writes are not atomic. A failure between them can leave ``frames`` rewritten
    (even empty) while ``events`` and ``drives`` still hold the previous run; rerunning repairs
    it, since ``raw_frames`` is durable and the run is a full recompute.

    With ``incremental`` only the drives `vl.etl.incremental.plan` reports as changed or removed
    are processed (see the module docstring); a full run happens when it reports ``None``. Every
    run ends by saving the state the next incremental run compares with.

    Args:
        spark: The Spark session.
        warehouse_dir: Warehouse directory that holds the tables.
        cfg: Pipeline thresholds, passed to the pandas functions.
        incremental: Recompute only what changed since the last run.

    Returns:
        The number of rows in each table after the run, keyed by ``frames``, ``events`` and
        ``drives``.
    """
    todo = plan(Path(warehouse_dir), cfg) if incremental else None
    if todo is None:
        _full_run(spark, warehouse_dir, cfg)
    elif todo.changed or todo.removed:
        _incremental_run(spark, Path(warehouse_dir), cfg, todo)
    save_state(Path(warehouse_dir), cfg)
    return {table: _read_table(spark, warehouse_dir, table).count() for table in OUTPUTS}


def _full_run(spark: SparkSession, warehouse_dir: str, cfg: Config) -> None:
    """Recompute and overwrite all three tables.

    Args:
        spark: The Spark session.
        warehouse_dir: Warehouse directory.
        cfg: Pipeline thresholds.
    """
    raw = _read_raw(spark, warehouse_dir)
    _write_frames(raw.groupBy(KEY).applyInPandas(_frames_of(cfg), FRAMES_DDL), warehouse_dir)
    frames = _read_table(spark, warehouse_dir, "frames").groupBy(KEY)
    _write_flat(frames.applyInPandas(_events_of(cfg), EVENTS_DDL), warehouse_dir, "events")
    _write_flat(frames.applyInPandas(_drive_of(cfg), DRIVES_DDL), warehouse_dir, "drives")


def _partitions(spark: SparkSession, table: Path, names: list[str], ddl: str) -> DataFrame:
    """Read some partitions of a table partitioned by ``drive_id``.

    Args:
        spark: The Spark session.
        table: The table directory.
        names: Partition directory names (``drive_id=...``).
        ddl: The table schema.

    Returns:
        Their rows, with ``drive_id`` from the directory names; empty when ``names`` is empty.
    """
    paths = [str(table / name) for name in names if (table / name).is_dir()]
    if not paths:
        return spark.createDataFrame([], ddl)
    return spark.read.schema(ddl).option("basePath", str(table)).parquet(*paths)


def _merge_flat(
    spark: SparkSession, root: Path, name: str, drop: list[str], fresh: DataFrame
) -> None:
    """Replace some drives' rows of a flat table (``events`` or ``drives``).

    The merged table is written to ``<name>.next`` first, because Spark cannot overwrite a
    directory it is reading. The live directory is then renamed out of the way and the new one
    renamed into its place: two back-to-back directory renames, each an atomic, near-instant
    metadata operation, instead of deleting the live directory's files (which can take as long as
    the table has rows) before the swap. A reader landing in the instant between the two renames
    still sees no table; removing that risks a stable pointer (a symlink, or a real table format
    such as Delta or Iceberg, already on the roadmap). The old directory is deleted only once the
    new one is already serving reads at the canonical path, so a crash between the two renames
    leaves no table on disk at all rather than a half-deleted one; the next run's `plan` already
    treats a missing output table as a reason for a full recompute, so this is self-healing.

    Args:
        spark: The Spark session.
        root: Warehouse directory.
        name: ``events`` or ``drives``.
        drop: Drive ids whose old rows go.
        fresh: The new rows of the recomputed drives.
    """
    kept = _read_table(spark, str(root), name).where(~F.col(KEY).isin(drop))
    staging = root / f"{name}.next"
    kept.unionByName(fresh).write.mode("overwrite").parquet(str(staging))
    discard = root / f"{name}.old"
    if discard.exists():
        shutil.rmtree(discard)  # a leftover from an earlier run that crashed before this point
    (root / name).rename(discard)
    staging.rename(root / name)
    shutil.rmtree(discard)


def _incremental_run(spark: SparkSession, root: Path, cfg: Config, todo: Plan) -> None:
    """Recompute the changed drives and remove the vanished ones.

    Args:
        spark: The Spark session.
        root: Warehouse directory.
        cfg: Pipeline thresholds.
        todo: The changed and removed partitions.
    """
    for partition in (*todo.changed, *todo.removed):
        shutil.rmtree(root / "frames" / partition, ignore_errors=True)
    raw = _partitions(spark, root / "raw_frames", todo.changed, RAW_FRAMES_DDL)
    raw.groupBy(KEY).applyInPandas(_frames_of(cfg), FRAMES_DDL).write.mode("append").partitionBy(
        KEY
    ).parquet(str(root / "frames"))
    frames = _partitions(spark, root / "frames", todo.changed, FRAMES_DDL).groupBy(KEY)
    drop = [drive_id_of(p) for p in (*todo.changed, *todo.removed)]
    _merge_flat(spark, root, "events", drop, frames.applyInPandas(_events_of(cfg), EVENTS_DDL))
    _merge_flat(spark, root, "drives", drop, frames.applyInPandas(_drive_of(cfg), DRIVES_DDL))


def _warehouse_problem(warehouse: Path) -> str | None:
    """Describe why a path cannot be a warehouse directory.

    Args:
        warehouse: The ``--warehouse`` argument.

    Returns:
        A one-line message when the path or one of its parents is an existing file, else
        ``None``.
    """
    blocking = next((p for p in (warehouse, *warehouse.parents) if p.is_file()), None)
    return None if blocking is None else f"warehouse {warehouse}: {blocking} is a file"


def _no_raw_notice(warehouse: Path) -> str | None:
    """Describe an input without data, which is legal but usually a mistake.

    Args:
        warehouse: The warehouse directory.

    Returns:
        A one-line notice when ``raw_frames`` has no Parquet file, else ``None``.
    """
    if any((warehouse / "raw_frames").rglob("*.parquet")):
        return None
    return f"no raw_frames data under {warehouse}; the three tables will be empty"


def _config_or_report(warehouse: Path) -> Config | None:
    """Check the warehouse path and load the ``VL_*`` thresholds.

    Args:
        warehouse: The warehouse directory argument.

    Returns:
        The config, or ``None`` after printing one ``error:`` line on stderr (an unusable
        warehouse path, or an invalid ``VL_*`` value).
    """
    try:
        if (problem := _warehouse_problem(warehouse)) is not None:
            raise ValueError(problem)
        return load_config()
    except ValueError as exc:
        print(f"error: {' '.join(str(exc).split())}", file=sys.stderr)
        return None


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the transform command line.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        ``warehouse`` (a path) and ``incremental`` (a flag).
    """
    parser = argparse.ArgumentParser(description="Recompute frames, events and drives.")
    parser.add_argument("--warehouse", required=True, type=Path, help="warehouse directory")
    parser.add_argument("--incremental", action="store_true", help="recompute changed drives only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the transform command line (``--warehouse DIR [--incremental]``).

    Builds the session with ``get_spark`` and calls `run_transform` on the warehouse, with the
    thresholds of ``load_config`` (defaults plus ``VL_<FIELD>`` overrides). Prints the row
    counts. A missing or empty ``raw_frames`` is legal (the tables are cleared, exit ``0``) but
    prints a one-line ``notice:`` on stderr.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code: ``0`` on success, ``2`` when the warehouse path is (or sits
        below) an existing file or a ``VL_*`` environment value is invalid; the reason is one
        ``error:`` line on stderr.
    """
    args = _parse_args(argv)
    warehouse = args.warehouse
    if (cfg := _config_or_report(warehouse)) is None:
        return 2
    if (notice := _no_raw_notice(warehouse)) is not None:
        print(f"notice: {notice}", file=sys.stderr)
    spark = get_spark("vl-transform")
    try:
        counts = run_transform(spark, str(warehouse), cfg, args.incremental)
    finally:
        spark.stop()
    print(", ".join(f"{table}: {n}" for table, n in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
