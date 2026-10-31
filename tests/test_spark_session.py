"""Tests for the Spark session factory ``vl.spark.get_spark`` (no network, no Kafka).

The Kafka-package test reads PySpark's private ``SparkSession.Builder._options`` (valid for
the pinned PySpark 3.5.8); it is the only network-free way to see the requested packages.
"""

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from pyspark.sql import SparkSession

from vl.spark import get_spark

KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8"


def _capture_builder_options(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Make ``SparkSession.Builder.getOrCreate`` record its options instead of building.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        A dict that holds the builder options after ``get_spark`` ran.
    """
    captured: dict[str, Any] = {}

    def fake_get_or_create(self: SparkSession.Builder) -> MagicMock:
        captured.update(self._options)
        return MagicMock()

    monkeypatch.setattr(SparkSession.Builder, "getOrCreate", fake_get_or_create)
    return captured


def test_get_spark_settings(spark: SparkSession) -> None:
    assert spark.conf.get("spark.sql.shuffle.partitions") == "2"
    assert spark.conf.get("spark.sql.sources.partitionColumnTypeInference.enabled") == "false"
    assert spark.conf.get("spark.sql.ansi.enabled") == "false"
    assert spark.sparkContext.getConf().get("spark.ui.enabled") == "false"
    packages = spark.sparkContext.getConf().get("spark.jars.packages", "")
    assert "kafka" not in packages


def test_get_spark_adds_kafka_package_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_builder_options(monkeypatch)
    # Isolate from the live shared session: this test is about the builder options only.
    monkeypatch.setattr(SparkSession, "getActiveSession", classmethod(lambda cls: None))
    get_spark("no-kafka")
    assert "kafka" not in str(captured.get("spark.jars.packages", ""))
    captured.clear()
    get_spark("with-kafka", with_kafka=True)
    assert KAFKA_PACKAGE in str(captured["spark.jars.packages"]).split(",")


def test_python_workers_use_the_driver_interpreter(spark: SparkSession) -> None:
    # Passes through the environment pin in tests/conftest.py whatever get_spark does; kept
    # as a guard that the workers never fall back to the system python3.
    workers = spark.sparkContext.parallelize(range(2), 2).map(lambda _: sys.executable)
    assert set(workers.collect()) == {sys.executable}
