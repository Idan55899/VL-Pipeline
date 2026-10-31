"""End-to-end Kafka run: producer CLI, Kafka ingest, transform, validator (needs `make kafka-up`).

Run with ``make sanity-kafka``. It needs its own Spark session with the Kafka connector, so it
must not share a process with a test that already created a session without it.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from kafka_helpers import fresh_topic
from test_sanity_phase2 import expected_from_jsonl
from vl.etl.ingest import run_ingest
from vl.etl.transform import run_transform
from vl.kafka_io import BOOTSTRAP
from vl.mock.producer import main as produce
from vl.validate import main as validate
from vl.warehouse import Warehouse

N_DRIVES, N_FRAMES = 3, 1000  # realism needs at least 1000 frames per drive


@pytest.fixture
def topic() -> str:
    """Skip the test unless a broker is reachable; give a fresh topic (before Spark starts)."""
    return fresh_topic()


@pytest.fixture
def kafka_spark(topic: str) -> Iterator[SparkSession]:
    """Provide a Spark session with the Kafka connector (downloaded on first use)."""
    from vl.spark import get_spark  # lazy: the connector download happens only here

    session = get_spark("vl-e2e", with_kafka=True)
    yield session
    session.stop()


def count(warehouse: Warehouse, table: str) -> int:
    """Count the rows of a warehouse table."""
    return int(warehouse.query(f"SELECT COUNT(*) AS n FROM {table}")["n"][0])


def produce_demo(fleet: Path, topic: str) -> tuple[int, int]:
    """Publish the demo fleet; return (messages sent, malformed messages) from the files."""
    args = ["--drives", str(N_DRIVES), "--frames", str(N_FRAMES), "--seed", "0"]
    args += ["--faults", "demo", "--bootstrap", BOOTSTRAP, "--topic", topic]
    assert produce([*args, "--out-dir", str(fleet)]) == 0
    sent = len((fleet / "frames.jsonl").read_text().splitlines())
    labels = json.loads((fleet / "labels.json").read_text())["labels"]
    return sent, sum(label["kind"] == "malformed" for label in labels)


@pytest.mark.integration
@pytest.mark.sanity
def test_kafka_pipeline_end_to_end(tmp_path: Path, topic: str, kafka_spark: SparkSession) -> None:
    """Demo fleet through Kafka: counts, rejects and the validator agree with the truth."""
    fleet, wh = tmp_path / "mock", tmp_path / "warehouse"
    sent, malformed = produce_demo(fleet, topic)
    run_ingest(kafka_spark, BOOTSTRAP, topic, str(wh), str(tmp_path / "checkpoint"))
    run_transform(kafka_spark, str(wh))
    warehouse = Warehouse(str(wh))
    assert malformed == 1 and sent == N_DRIVES * N_FRAMES  # the drop and the duplicate cancel
    assert count(warehouse, "rejects") == malformed
    assert count(warehouse, "raw_frames") == sent - malformed
    expected = expected_from_jsonl(fleet / "frames.jsonl")
    assert count(warehouse, "drives") == len(expected) == N_DRIVES
    assert count(warehouse, "frames") == sum(len(t.frames) for t in expected.values())
    source, labels = str(wh / "raw_frames"), str(fleet / "labels.json")
    assert validate(["--source", source, "--labels", labels, "--realism"]) == 0
