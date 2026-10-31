"""Failure behaviour of the Kafka CLIs: one `error:` line, exit 1, no traceback, no hang."""

import pytest
from kafka.errors import KafkaError

from kafka_helpers import FakeProducer
from vl.etl import ingest
from vl.kafka_io import KafkaUnavailableError, TopicNotFoundError
from vl.mock import producer

PRODUCE = ["--drives", "2", "--frames", "50", "--seed", "0"]


class RaisingProducer(FakeProducer):
    """A producer whose send raises ``KafkaError``."""

    def send(self, topic: str, *, key: bytes, value: bytes) -> object:
        """Fail like a producer that lost its broker."""
        raise KafkaError("broker lost")


class FailedFuture:
    """A send future that completed with an error."""

    def get(self, timeout: float | None = None) -> object:
        """Raise the delivery error."""
        raise KafkaError("not enough replicas")


class FailingFutureProducer(FakeProducer):
    """A producer whose sends are accepted but fail to deliver."""

    def send(self, topic: str, *, key: bytes, value: bytes) -> object:
        """Return a future that fails."""
        return FailedFuture()


def only_error_line(capsys: pytest.CaptureFixture[str], case: str) -> str:
    """Return the single stderr line, which must start with ``error:`` (no traceback)."""
    lines = [x for x in capsys.readouterr().err.splitlines() if x.strip()]
    assert len(lines) == 1 and lines[0].startswith("error: "), (case, lines)
    return lines[0]


def test_producer_main_reports_kafka_failures_in_one_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing send, a failed delivery and an unreachable broker all exit 1 cleanly."""
    for fake in (RaisingProducer(), FailingFutureProducer()):
        assert producer.main(PRODUCE, producer=fake) == 1, type(fake).__name__
        only_error_line(capsys, type(fake).__name__)

    def down(bootstrap: str) -> FakeProducer:
        raise KafkaUnavailableError(bootstrap)

    monkeypatch.setattr("vl.kafka_io.build_producer", down)
    assert producer.main([*PRODUCE, "--bootstrap", "h:1"]) == 1
    assert only_error_line(capsys, "down") == "error: cannot reach Kafka at h:1"


def test_ingest_main_prechecks_the_broker_and_the_topic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unreachable broker or a missing topic exits 1 before Spark starts."""
    errors: list[Exception] = [KafkaUnavailableError("h:1"), TopicNotFoundError("t")]
    expected = [
        "error: cannot reach Kafka at h:1",
        "error: topic t not found (run make produce first)",
    ]
    for error, line in zip(errors, expected):

        def probe(bootstrap: str, topic: str, err: Exception = error) -> None:
            raise err

        monkeypatch.setattr(ingest, "check_broker", probe)
        assert ingest.main(["--kafka", "--bootstrap", "h:1", "--topic", "t"]) == 1, line
        assert only_error_line(capsys, line) == line
