"""Shared helpers of the Kafka tests: fakes for unit tests, broker access for integration tests."""

import socket
import uuid
from typing import Any

import pytest

from vl.kafka_io import BOOTSTRAP

Sent = tuple[str, bytes, bytes]


class FakeClock:
    """A manual clock whose ``sleep`` advances it; send times are recorded by the producer."""

    def __init__(self) -> None:
        """Start at time zero with no recorded sleeps."""
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        """Return the current fake time in seconds."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Record a sleep and advance the fake time by it."""
        self.sleeps.append(seconds)
        self.now += seconds


class FakeProducer:
    """A ``ProducerLike`` that records every message instead of sending it."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        """Create an empty producer; with a clock, the send times are recorded too."""
        self.sent: list[Sent] = []
        self.times: list[float] = []
        self.flushes = 0
        self._clock = clock

    def send(self, topic: str, *, key: bytes, value: bytes) -> object:
        """Record one message (and the fake time it was sent at)."""
        self.sent.append((topic, key, value))
        if self._clock is not None:
            self.times.append(self._clock.now)
        return None

    def flush(self) -> None:
        """Count a flush."""
        self.flushes += 1


def fresh_topic() -> str:
    """Skip the test unless a broker answers on ``BOOTSTRAP``; return an unused topic name.

    Returns:
        A new topic name (the broker auto-creates it on first use).
    """
    host, _, port = BOOTSTRAP.rpartition(":")
    try:
        socket.create_connection((host, int(port)), timeout=1).close()
    except OSError:
        pytest.skip(f"no Kafka broker reachable at {BOOTSTRAP} (run `make kafka-up`)")
    return f"vl.test.{uuid.uuid4().hex[:12]}"


def consume(topic: str, count: int, timeout_s: float = 20.0) -> list[Any]:
    """Read ``count`` records of a topic from the beginning.

    Args:
        topic: The topic to read.
        count: Number of records to wait for.
        timeout_s: Give up when no record arrives for this long.

    Returns:
        The consumed records (fewer than ``count`` if the timeout hit).
    """
    from kafka import KafkaConsumer  # lazy: unit tests need no kafka-python import

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=BOOTSTRAP,
        auto_offset_reset="earliest",
        group_id=None,
        consumer_timeout_ms=int(timeout_s * 1000),
    )
    records: list[Any] = []
    try:
        for record in consumer:
            records.append(record)
            if len(records) == count:
                break
    finally:
        consumer.close()
    return records
