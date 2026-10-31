"""Kafka integration: publish frames to a real broker and read them back (needs `make kafka-up`)."""

import pytest

from kafka_helpers import consume, fresh_topic
from sim_helpers import make_frames
from vl.kafka_io import BOOTSTRAP, decode_frame, publish_drive


@pytest.mark.integration
def test_publish_and_consume() -> None:
    """Five frames published with the real producer come back keyed, in order, unchanged."""
    topic = fresh_topic()
    frames = make_frames(5, "drive-007")
    assert publish_drive([*frames], BOOTSTRAP, topic) == 5
    records = consume(topic, 5)
    assert len(records) == 5
    assert [decode_frame(r.value) for r in records] == frames
    assert {r.key for r in records} == {b"drive-007"}
