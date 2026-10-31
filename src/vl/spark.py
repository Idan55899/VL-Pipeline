"""Spark session factory shared by the ingest, transform and test code.

Role in the pipeline: every Spark job (streaming ingest, batch transform) and the test
suite build their session here, so the settings stay identical everywhere.
"""

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8"


def _src_root() -> str:
    """Locate the directory that holds the ``vl`` package (the ``src`` directory).

    Returns:
        Absolute path of the parent of the installed ``vl`` package.
    """
    return str(Path(__file__).resolve().parent.parent)


def _pin_python_and_path(src_root: str) -> None:
    """Export the interpreter pin and the import path to the process environment.

    The Spark JVM and its Python workers inherit the environment of the process that starts
    the session, so this must run before the first session is created.

    Args:
        src_root: Directory that holds the ``vl`` package.
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if src_root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([src_root, *parts])


def _require_kafka_package_if_session_exists(with_kafka: bool) -> None:
    """Refuse to reuse a live session that lacks the requested Kafka package.

    ``getOrCreate`` returns a running session unchanged, silently ignoring a new
    ``spark.jars.packages``; the Kafka source would then fail much later with a confusing error.

    Args:
        with_kafka: Whether the caller needs the Kafka connector.

    Raises:
        RuntimeError: If ``with_kafka`` is true and the active session has no Kafka package.
    """
    active = SparkSession.getActiveSession()
    if not with_kafka or active is None:
        return
    packages = active.sparkContext.getConf().get("spark.jars.packages", "")
    if KAFKA_PACKAGE not in packages.split(","):
        raise RuntimeError(
            "An existing Spark session was created without the Kafka package "
            f"({KAFKA_PACKAGE}); stop it with spark.stop() or start a new process, then call "
            "get_spark(..., with_kafka=True) before any other get_spark call."
        )


def get_spark(app_name: str, master: str = "local[2]", with_kafka: bool = False) -> SparkSession:
    """Build (or reuse) the Spark session with the pipeline's settings.

    Sets ``spark.sql.shuffle.partitions=2`` (a local demo setting: at scale it follows the
    data size, with adaptive query execution, on by default in Spark 3.5, coalescing
    partitions), ``spark.sql.parquet.compression.codec=zstd`` (smaller files than the default
    Snappy at similar read speed; DuckDB and PyArrow read it),
    ``spark.sql.sources.partitionColumnTypeInference.enabled=false`` (so a numeric-looking
    ``drive_id`` stays a string), ``spark.ui.enabled=false`` and
    ``spark.sql.ansi.enabled=false`` (the ingest reads ``array[0]`` of an empty array as
    null, which ANSI mode would turn into an error), and pins the Python
    workers to the driver interpreter. The directory of the ``vl`` package is put on the
    workers' ``PYTHONPATH`` (environment and ``spark.executorEnv.PYTHONPATH``), so
    ``applyInPandas`` functions can import ``vl``. The Kafka connector package
    ``org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8`` is added through
    ``spark.jars.packages`` only when ``with_kafka`` is true, so unit tests need no network.

    Args:
        app_name: Spark application name.
        master: Spark master URL.
        with_kafka: Whether to add the Kafka connector package.

    Returns:
        The configured session.

    Raises:
        RuntimeError: If ``with_kafka`` is true but a live session without the Kafka package
            already exists (Spark would silently reuse it).
    """
    _require_kafka_package_if_session_exists(with_kafka)
    src_root = _src_root()
    _pin_python_and_path(src_root)
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
    )
    if with_kafka:
        builder = builder.config("spark.jars.packages", KAFKA_PACKAGE)
    return builder.getOrCreate()
