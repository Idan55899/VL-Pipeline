"""Streaming ingest: Kafka (or files) to the ``raw_frames`` and ``rejects`` Parquet tables.

Role in the pipeline: the first Spark stage. It parses the JSON frame messages with the
strict message contract, reduces each point cloud to its centroid, appends valid frames to
``raw_frames`` (partitioned by ``drive_id``) and sends malformed messages to ``rejects``
with the error code ``VL-005``.

Two message versions are accepted. v1 (``schema_version`` absent, null or 1) carries the
``points``. v2 (``schema_version`` 2, the "claim check" of `vl.claim_check`) carries no points:
``centroid_m`` (three finite numbers, absent when ``n_points`` is 0), ``n_points`` (0 to
``MAX_POINTS``) and the string ``cloud_ref``; its ``points`` field, if any, is ignored. Any other
``schema_version`` (``3``, ``"2"``, ``2.0``, ``true``) is malformed.

Strict parse design. Spark's ``from_json`` is native and fast but partly lenient: with a
string type it accepts a number, boolean, object or array as text, it reads only the first
JSON value of a line (trailing text is ignored) and it accepts single quotes. The contract
is enforced without any Python UDF, using only Spark SQL expressions:

* The line is wrapped as ``"[" + value + ",null]"`` and parsed as an array of structs; it
  is exactly one JSON value if and only if that array has exactly two elements (the
  message and our ``null``). ``from_json`` stops at the first closing ``]``, so a ``]``
  inside the line would end the array early and hide whatever follows it. A second parse of
  ``"[" + value + ",null,null]"`` must therefore give exactly three elements: an early
  close yields the same array for both wrappers (same prefix), so it can never satisfy both
  counts. Trailing text, concatenated objects, comma lists and a non-object root are all
  rejected. ``allowSingleQuotes`` and ``allowNonNumericNumbers`` are switched off.
* Typed fields are strict in Spark 3.5: ``frame_idx`` as ``long`` rejects ``1.0``, ``"5"``
  and values beyond 64 bits (the field becomes null); ``points`` as ``array<array<long>>``
  rejects float, string and boolean coordinates (the whole array becomes null). What typing
  cannot see is checked explicitly: every point is a non-null triple of non-null integers,
  and ``ts`` is a finite double (Spark reads the strings ``"NaN"`` and ``"Infinity"`` as
  doubles, a JSON number never is one).
* The string fields (``drive_id`` and the optional metadata) are checked with four extra
  ``from_json`` probes that type the same fields as ``double``, ``boolean``,
  ``map<string,string>`` and ``array<string>``. A field that parses under any probe was not
  a JSON string, so the line is rejected. An explicit ``null`` parses as null under every
  probe and counts as absent.

Why not a Python UDF: it would ship every message, point arrays included, to a Python
worker and back, which costs a serialization round trip per row; the probes stay inside the
JVM and skip over the point cloud. The centroid is plain Spark SQL: one ``aggregate`` pass
that sums the coordinates as ``double`` (so integer centimeters near the 32-bit limit
cannot overflow) and divides by ``100 * n_points``.

Usage: ``python -m vl.etl.ingest --source-dir DIR`` (files) or ``python -m vl.etl.ingest
--kafka --bootstrap localhost:9092 --topic vl.frames`` (Kafka), both with ``--warehouse
data/warehouse [--checkpoint DIR]``.

Known limitations (by design or inherited from Spark; no code handles them):

* Duplicate JSON keys: Spark keeps the last value of a typed field, and a later invalid
  duplicate is ignored, so the outcome depends on key order and type. The message contract
  is silent about duplicates (release-notes item).
* A JSON number beyond the double range (such as ``1e999``) in a string field is read as
  text instead of being rejected.
* ``write_batch`` is idempotent per micro-batch: a batch Spark replays after a crash replaces
  the files of its earlier attempt instead of adding its rows a second time.
* A missing source directory makes ``run_ingest_files`` raise Spark's ``PATH_NOT_FOUND``
  ``AnalysisException``.
"""

import argparse
import shutil
import sys
import uuid
from collections.abc import Sequence
from functools import reduce
from operator import or_
from pathlib import Path

from pyspark.errors import AnalysisException
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from vl.errors import ErrorCode
from vl.kafka_io import BOOTSTRAP, TOPIC, KafkaUnavailableError, TopicNotFoundError, check_broker
from vl.schemas import MAX_POINTS, MESSAGE_STRING_FIELDS, RAW_FRAMES_COLUMNS, SCHEMA_V2
from vl.spark import get_spark

REJECT_CODE = ErrorCode.MALFORMED_MESSAGE.value
# Largest micro-batch read from Kafka: a backlog is split into bounded batches (memory stays flat)
# instead of one batch holding everything; availableNow still drains the whole backlog.
MAX_OFFSETS_PER_TRIGGER = 100_000
# Where a micro-batch is written before it is moved into the tables (see `write_batch`).
STAGING_DIR = "_staging"
RUN_ID_FILE = "vl_run_id"
_MESSAGE_SCHEMA = (
    "array<struct<drive_id:string, vehicle_id:string, vehicle_kind:string, "
    "drive_start_utc:string, frame_idx:long, ts:double, points:array<array<long>>, "
    "schema_version:long, centroid_m:array<double>, n_points:long, cloud_ref:string>>"
)
# The raw tokens of v2 fields whose typed parse turns a wrong JSON type into null (to reject
# a schema_version "2", 2.0 or true, and a centroid_m that is not an array).
_TOKEN_SCHEMA = "schema_version string, centroid_m string"
_JSON_OPTIONS = {"allowSingleQuotes": "false", "allowNonNumericNumbers": "false"}
_COUNT_SCHEMA = "array<struct<drive_id:string>>"
_NON_STRING_PROBES = ("double", "boolean", "map<string,string>", "array<string>")


def _probe_hit(field: Column, type_name: str) -> Column:
    """Tell whether a probe parse found a non-string token in a field.

    Args:
        field: The field as parsed under the probe type.
        type_name: The probe type.

    Returns:
        True (never null) when the probe parsed a value. Under ``double`` only finite
        values count, because Spark also reads the strings ``"NaN"`` and ``"Infinity"`` as
        doubles.
    """
    hit = F.abs(field) < float("inf") if type_name == "double" else field.isNotNull()
    return F.coalesce(hit, F.lit(False))


def _non_string_probe(value: Column, type_name: str) -> Column:
    """Check whether any string field holds a token that parses as ``type_name``.

    Args:
        value: The raw message column.
        type_name: The probe type.

    Returns:
        True when at least one of the string fields is not a JSON string.
    """
    schema = ", ".join(f"{field} {type_name}" for field in MESSAGE_STRING_FIELDS)
    probe = F.from_json(value, schema, _JSON_OPTIONS)
    return reduce(or_, [_probe_hit(probe[field], type_name) for field in MESSAGE_STRING_FIELDS])


def _has_non_string_field(value: Column) -> Column:
    """Check whether a string field of the message is a number, boolean, object or array.

    Args:
        value: The raw message column.

    Returns:
        True when the message violates "these fields are JSON strings".
    """
    return reduce(or_, [_non_string_probe(value, name) for name in _NON_STRING_PROBES])


def _points_are_triples(points: Column) -> Column:
    """Check that every point is a non-null triple of non-null integers.

    Args:
        points: The ``array<array<long>>`` column (null when the JSON typing failed).

    Returns:
        True for a valid, possibly empty, cloud; null or False otherwise.
    """
    return F.forall(
        points,
        lambda point: (
            point.isNotNull()
            & (F.size(point) == 3)
            & F.forall(point, lambda coordinate: coordinate.isNotNull())
        ),
    )


def _is_single_value(value: Column, frames: Column) -> Column:
    """Check that the line is exactly one JSON value (see the module docstring).

    Args:
        value: The raw message column.
        frames: The line wrapped with one trailing ``null`` and parsed as message structs.

    Returns:
        True when the first wrapper gives two elements and the second wrapper three.
    """
    counted = F.from_json(_wrap(value, ",null,null]"), _COUNT_SCHEMA, _JSON_OPTIONS)
    return (F.size(frames) == 2) & (F.size(counted) == 3)


def _wrap(value: Column, tail: str) -> Column:
    """Wrap a line into a JSON array text.

    Args:
        value: The raw message column.
        tail: Text appended after the line, including the closing bracket.

    Returns:
        ``"[" + value + tail``.
    """
    return F.concat(F.lit("["), value, F.lit(tail))


def _is_valid(value: Column, frames: Column) -> Column:
    """Build the condition "the line satisfies the strict message contract".

    Args:
        value: The raw message column.
        frames: The line wrapped with one trailing ``null`` and parsed as message structs.

    Returns:
        A boolean column that is never null.
    """
    frame = frames[0]
    required = (
        frame["drive_id"].isNotNull()
        & frame["frame_idx"].isNotNull()
        & (F.abs(frame["ts"]) < float("inf"))
        & _version_ok(value, frame)
        & _cloud_ok(value, frame)
    )
    single = _is_single_value(value, frames)
    return F.coalesce(single & required & ~_has_non_string_field(value), F.lit(False))


def _is_v2(frame: Column) -> Column:
    """Tell whether a parsed message is version 2.

    Args:
        frame: The parsed message struct.

    Returns:
        True (never null) when ``schema_version`` is the integer 2.
    """
    return F.coalesce(frame["schema_version"] == SCHEMA_V2, F.lit(False))


def _version_ok(value: Column, frame: Column) -> Column:
    """Check ``schema_version``: absent or null (v1), or the integer 1 or 2.

    Typed as ``long`` the field rejects ``2.0``, ``"2"`` and ``true`` by becoming null, so the
    raw token is read again as a string: a null long with a non-null token was not an integer.

    Args:
        value: The raw message column.
        frame: The parsed message struct.

    Returns:
        A boolean column (null only where the line is not a JSON object).
    """
    version, token = frame["schema_version"], _token(value, "schema_version")
    return version.isin(1, SCHEMA_V2) | (version.isNull() & token.isNull())


def _token(value: Column, field: str) -> Column:
    """Read a field's raw JSON token as text, whatever its JSON type.

    Args:
        value: The raw message column.
        field: ``schema_version`` or ``centroid_m``.

    Returns:
        The token's text; null when the field is absent or JSON ``null``.
    """
    return F.from_json(value, _TOKEN_SCHEMA, _JSON_OPTIONS)[field]


def _v2_cloud_ok(value: Column, frame: Column) -> Column:
    """Check the cloud fields of a v2 message.

    Args:
        value: The raw message column (an empty cloud needs ``centroid_m`` truly absent: a
            non-array token, such as a string, is also null once typed).
        frame: The parsed message struct.

    Returns:
        True when ``cloud_ref`` is present (its string type is checked by the probes),
        ``0 <= n_points <= MAX_POINTS``, and ``centroid_m`` is three finite numbers, or null when
        ``n_points`` is 0.
    """
    n_points, centroid = frame["n_points"], frame["centroid_m"]
    finite = F.forall(centroid, lambda v: F.abs(v) < float("inf"))
    full = (n_points > 0) & (n_points <= MAX_POINTS) & (F.size(centroid) == 3) & finite
    empty = (n_points == 0) & _token(value, "centroid_m").isNull()
    return frame["cloud_ref"].isNotNull() & (empty | full)


def _cloud_ok(value: Column, frame: Column) -> Column:
    """Check the cloud of a message: v1 ``points`` triples, or the v2 cloud fields.

    Args:
        value: The raw message column.
        frame: The parsed message struct.

    Returns:
        A boolean column (null where a field could not be read).
    """
    return F.when(_is_v2(frame), _v2_cloud_ok(value, frame)).otherwise(
        _points_are_triples(frame["points"])
    )


def _centroid(points: Column) -> Column:
    """Compute the cloud centroid in meters with one Spark SQL ``aggregate`` pass.

    Args:
        points: The ``array<array<long>>`` cloud in integer centimeters.

    Returns:
        A struct ``(x, y, z)`` of doubles, null for an empty cloud.
    """
    zero = F.struct(F.lit(0.0).alias("x"), F.lit(0.0).alias("y"), F.lit(0.0).alias("z"))
    total = F.aggregate(
        points,
        zero,
        lambda acc, p: F.struct(
            (acc["x"] + p[0]).alias("x"),
            (acc["y"] + p[1]).alias("y"),
            (acc["z"] + p[2]).alias("z"),
        ),
    )
    scale = F.size(points) * 100.0
    centroid = F.struct(*[(total[axis] / scale).alias(axis) for axis in "xyz"])
    return F.when(F.size(points) > 0, centroid)


def _classify(raw: DataFrame) -> DataFrame:
    """Parse every line once and compute all output columns plus the validity flag.

    Args:
        raw: DataFrame with a string column ``value``.

    Returns:
        ``value``, the boolean ``ok`` and the ``RAW_FRAMES_COLUMNS`` (meaningful where ``ok``).
    """
    value = F.col("value")
    frames = F.from_json(_wrap(value, ",null]"), _MESSAGE_SCHEMA, _JSON_OPTIONS)
    staged = raw.select(value, _is_valid(value, frames).alias("ok"), frames[0].alias("frame"))
    staged = staged.withColumn("centroid", _centroid(F.col("frame.points")))
    frame, centroid, v2 = F.col("frame"), F.col("centroid"), _is_v2(F.col("frame"))
    columns = [frame[name].alias(name) for name in RAW_FRAMES_COLUMNS[:6]]
    n_points = F.when(v2, frame["n_points"].cast("int")).otherwise(F.size(frame["points"]))
    columns.append(n_points.alias("n_points"))
    columns += [
        F.when(v2, frame["centroid_m"][i]).otherwise(centroid[axis]).alias(axis)
        for i, axis in enumerate("xyz")
    ]
    return staged.select("value", "ok", *columns)


def _split(classified: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Split classified lines into the ``raw_frames`` and ``rejects`` DataFrames.

    Args:
        classified: Output of `_classify`.

    Returns:
        The pair ``(valid, rejects)``.
    """
    valid = classified.where("ok").select(*RAW_FRAMES_COLUMNS)
    rejects = classified.where(~F.col("ok")).select(
        F.col("value").alias("payload"),
        F.lit(REJECT_CODE).alias("error_code"),
        F.current_timestamp().alias("ingested_at"),
    )
    return valid, rejects


def parse_messages(raw: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Split raw message strings into valid frames and rejected messages.

    ``valid`` matches ``RAW_FRAMES_DDL`` exactly (column order included): ``x, y, z`` are
    the cloud centroid in meters from integer-cm points (``mean / 100``) and ``n_points``
    is the point count. An empty cloud (``points: []``) is valid with ``n_points = 0`` and
    null ``x, y, z``. The message contract is strict: ``drive_id`` is a JSON string,
    ``frame_idx`` a JSON integer, ``ts`` a JSON number (integers allowed), ``points`` an
    array of arrays of exactly 3 JSON integers, and ``vehicle_id``, ``vehicle_kind`` and
    ``drive_start_utc`` JSON strings when present (null when absent). Unknown extra fields
    are ignored. Anything else goes to ``rejects`` (matching ``REJECTS_DDL``) with
    ``error_code = "VL-005"``, the original string as ``payload`` and the current timestamp
    as ``ingested_at``. The centroid uses Spark SQL higher-order functions, not a UDF; see
    the module docstring for how the strict contract is enforced.

    Args:
        raw: DataFrame with a string column ``value``.

    Returns:
        The pair ``(valid, rejects)``.
    """
    return _split(_classify(raw))


def _stage(df: DataFrame, path: Path, partition_by: list[str]) -> None:
    """Write a batch's rows to its staging directory, replacing an earlier attempt's files.

    Args:
        df: Rows of one table.
        path: The batch's staging directory for that table.
        partition_by: Partition columns (empty for an unpartitioned table).
    """
    if df.isEmpty():
        shutil.rmtree(path, ignore_errors=True)  # nothing to publish, not even a schema file
        return
    writer = df.write.mode("overwrite")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(str(path))


def _publish(staged: Path, table: Path, tag: str) -> None:
    """Move a batch's staged files into its table, replacing the batch's earlier files.

    Every published file is named ``<tag>-<Spark file name>``, so the files of an earlier
    attempt of the same batch (all of them, or the part a crash let through) are found by
    name and removed before the new ones move in. A move is a rename on the same disk.

    Args:
        staged: The batch's staging directory for this table (may be missing: no rows).
        table: The table directory.
        tag: The batch tag, unique per checkpoint and batch id.
    """
    for old in table.glob(f"**/{tag}-*.parquet"):
        old.unlink()
    for path in sorted(staged.rglob("*.parquet")) if staged.is_dir() else []:
        target = table / path.relative_to(staged).parent / f"{tag}-{path.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)


def write_batch(batch: DataFrame, batch_id: int, out_dir: str, run_id: str = "manual") -> None:
    """Parse one micro-batch and add the results to the Parquet tables, idempotently.

    ``valid`` goes to ``<out_dir>/raw_frames`` (partitioned by ``drive_id``) and ``rejects``
    to ``<out_dir>/rejects``. An empty batch writes nothing and does not fail. The parsed
    batch is cached while both tables are written, so every line is parsed once.

    Spark replays a micro-batch, with the same id and the same rows, when it fails before its
    checkpoint commit: a crash after some files were written, or after all of them. Plain
    appends would add the replayed rows a second time, and the transform would flag each copy
    VL-004 like a sensor duplicate. So both tables are first written to
    ``<out_dir>/_staging/<tag>/`` and then moved into place by `_publish`, which replaces any
    files an earlier attempt of the same batch left: a replay leaves exactly one copy.

    Args:
        batch: DataFrame with a string column ``value``.
        batch_id: Micro-batch id given by ``foreachBatch``.
        out_dir: Warehouse directory that holds the tables.
        run_id: Names the checkpoint the batch ids belong to (`_run_id`); batch ids restart
            at 0 with a new checkpoint, so the id alone does not name a batch.
    """
    tag = f"batch-{run_id}-{batch_id}"
    staging = Path(out_dir) / STAGING_DIR / tag
    classified = _classify(batch).persist()
    try:
        valid, rejects = _split(classified)
        _stage(valid, staging / "raw_frames", ["drive_id"])
        _stage(rejects, staging / "rejects", [])
    finally:
        classified.unpersist()
    for table in ("raw_frames", "rejects"):
        _publish(staging / table, Path(out_dir) / table, tag)
    if staging.exists():  # an empty batch stages nothing
        shutil.rmtree(staging)


def _run_id(checkpoint_dir: str) -> str:
    """Read, or create on first use, the id of a checkpoint's run.

    It lives inside the checkpoint directory, so deleting the checkpoint (``make kafka-reset``)
    starts a new run: batch 0 of the new run never replaces the files of the old run's batch 0.

    Args:
        checkpoint_dir: Directory of the streaming checkpoint.

    Returns:
        The run id (32 hex digits).
    """
    path = Path(checkpoint_dir) / RUN_ID_FILE
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(uuid.uuid4().hex)
    return path.read_text().strip()


def ingest_stream(stream: DataFrame, out_dir: str, checkpoint_dir: str) -> None:
    """Run the ingest over a streaming DataFrame until the available data is consumed.

    Uses ``trigger(availableNow=True)`` and ``foreachBatch`` calling `write_batch` with the
    checkpoint's run id (`_run_id`), keeps the checkpoint in ``checkpoint_dir`` and waits for
    completion.

    Args:
        stream: Streaming DataFrame with a string column ``value``.
        out_dir: Warehouse directory that holds the tables.
        checkpoint_dir: Directory of the streaming checkpoint.
    """
    run_id = _run_id(checkpoint_dir)

    def handle(batch: DataFrame, batch_id: int) -> None:
        write_batch(batch, batch_id, out_dir, run_id)

    query = (
        stream.writeStream.foreachBatch(handle)
        .option("checkpointLocation", checkpoint_dir)
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()


def run_ingest_files(
    spark: SparkSession, source_dir: str, out_dir: str, checkpoint_dir: str
) -> None:
    """Ingest text files from a directory (the Kafka-free source for tests and sanity runs).

    Reads ``spark.readStream.text(source_dir)`` into `ingest_stream`; with the same
    checkpoint semantics as the Kafka source, files are processed once.

    Args:
        spark: The Spark session.
        source_dir: Directory of files with one message per line.
        out_dir: Warehouse directory that holds the tables.
        checkpoint_dir: Directory of the streaming checkpoint.
    """
    ingest_stream(spark.readStream.text(source_dir), out_dir, checkpoint_dir)


def run_ingest(
    spark: SparkSession, bootstrap: str, topic: str, out_dir: str, checkpoint_dir: str
) -> None:
    """Ingest the frame messages of a Kafka topic.

    Reads the topic with ``startingOffsets=earliest``, at most ``MAX_OFFSETS_PER_TRIGGER``
    messages per micro-batch, and the ``value`` cast to string into
    `ingest_stream`. Its Kafka integration check belongs to the Phase 3 sanity test.

    Args:
        spark: A session built with ``with_kafka=True``.
        bootstrap: Kafka bootstrap servers.
        topic: Topic that carries the frame messages.
        out_dir: Warehouse directory that holds the tables.
        checkpoint_dir: Directory of the streaming checkpoint.
    """
    stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", MAX_OFFSETS_PER_TRIGGER)
        .load()
        .select(F.col("value").cast("string").alias("value"))
    )
    ingest_stream(stream, out_dir, checkpoint_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the ingest command line (pure: no Spark, no I/O).

    Options: ``--warehouse DIR`` (default ``data/warehouse``); ``--checkpoint DIR`` (default
    ``<warehouse>/_checkpoints/ingest``, joined with ``/``); exactly one source, either
    ``--source-dir DIR`` or ``--kafka``; with ``--kafka``, ``--bootstrap H:P`` (default
    ``vl.kafka_io.BOOTSTRAP``) and ``--topic T`` (default ``vl.kafka_io.TOPIC``).

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        A namespace with the attributes ``warehouse`` (str), ``checkpoint`` (str, the default
        resolved), ``source_dir`` (str or ``None``), ``kafka`` (bool), ``bootstrap`` (str) and
        ``topic`` (str).

    Raises:
        SystemExit: With status ``2`` (argparse) when no source or both sources are given, or
            an option is unknown or lacks its value.
    """
    parser = argparse.ArgumentParser(prog="vl.etl.ingest", description=__doc__)
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--checkpoint", default=None)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", default=None)
    source.add_argument("--kafka", action="store_true")
    parser.add_argument("--bootstrap", default=BOOTSTRAP)
    parser.add_argument("--topic", default=TOPIC)
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        args.checkpoint = f"{args.warehouse}/_checkpoints/ingest"
    return args


def _count_rows(spark: SparkSession, path: str) -> int:
    """Count the rows of a Parquet table, or 0 when nothing was written to it.

    Args:
        spark: The Spark session.
        path: Table directory.

    Returns:
        The number of rows.
    """
    try:
        return spark.read.parquet(path).count()
    except AnalysisException:
        return 0


def _table_counts(spark: SparkSession, warehouse: str) -> dict[str, int]:
    """Count the rows of the ``raw_frames`` and ``rejects`` tables.

    Args:
        spark: The Spark session.
        warehouse: Warehouse directory.

    Returns:
        Row counts by table name (0 for a table that does not exist yet).
    """
    return {t: _count_rows(spark, f"{warehouse}/{t}") for t in ("raw_frames", "rejects")}


def _run_selected_source(spark: SparkSession, args: argparse.Namespace) -> str:
    """Run the ingest over the source the arguments select.

    Args:
        spark: The Spark session.
        args: Parsed arguments (see `parse_args`).

    Returns:
        A short description of the source for the summary line.
    """
    if args.kafka:
        run_ingest(spark, args.bootstrap, args.topic, args.warehouse, args.checkpoint)
        return f"topic {args.topic} at {args.bootstrap}"
    run_ingest_files(spark, args.source_dir, args.warehouse, args.checkpoint)
    return f"directory {args.source_dir}"


def _broker_ready(bootstrap: str, topic: str) -> bool:
    """Check the broker and the topic; print one ``error:`` line to stderr when not ready.

    Args:
        bootstrap: Kafka bootstrap servers.
        topic: Topic that must exist.

    Returns:
        ``True`` when the broker answers and has the topic.
    """
    try:
        check_broker(bootstrap, topic)
    except (KafkaUnavailableError, TopicNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return False
    return True


def _describe(counts: dict[str, int]) -> str:
    """Format table counts as ``N raw_frames, M rejects``.

    Args:
        counts: Row counts with the keys ``raw_frames`` and ``rejects``.

    Returns:
        The text.
    """
    return f"{counts['raw_frames']} raw_frames, {counts['rejects']} rejects"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ingest command line.

    With ``--source-dir`` it calls `run_ingest_files` (session ``get_spark("vl-ingest")``);
    with ``--kafka`` it first checks (`check_broker`, a few seconds at most) that the broker
    answers and has the topic, then calls `run_ingest` (session ``get_spark("vl-ingest",
    with_kafka=True)``). Prints one summary line with the rows added by this run and the
    totals, and stops the session.

    Args:
        argv: Command line arguments (see `parse_args`); defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``1`` with one ``error: ...`` line on stderr when the broker cannot
        be reached or the topic is missing; bad arguments exit with status ``2`` (argparse).
    """
    args = parse_args(argv)
    if args.kafka and not _broker_ready(args.bootstrap, args.topic):
        return 1
    spark = get_spark("vl-ingest", with_kafka=args.kafka)
    try:
        before = _table_counts(spark, args.warehouse)
        origin = _run_selected_source(spark, args)
        after = _table_counts(spark, args.warehouse)
    finally:
        spark.stop()
    added = {t: after[t] - before[t] for t in after}
    print(f"ingested {origin}: added {_describe(added)} (totals: {_describe(after)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
