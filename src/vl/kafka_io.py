"""Kafka message encoding and publishing of frame messages.

Role in the pipeline: the producer side of the Kafka hop. It turns frame messages (dicts) into
the bytes that travel on the ``vl.frames`` topic and back, and publishes a drive's frames in
emission order with the ``drive_id`` as the message key, so one drive stays ordered in one
partition. The Spark ingest consumes the same topic.

Wire format: UTF-8 JSON, compact separators (``","`` and ``":"``), non-ASCII characters kept as
they are (``ensure_ascii=False``), keys in insertion order.
"""

import json
import os
import time
from collections.abc import Callable
from typing import Protocol, cast

from kafka import KafkaAdminClient, KafkaProducer
from kafka.errors import KafkaError

TOPIC = "vl.frames"
BOOTSTRAP = os.environ.get("VL_KAFKA_BOOTSTRAP", "localhost:9092")
CONNECT_TIMEOUT_MS = 3000  # give up on an unreachable broker after this long
_DELIVERY_TIMEOUT_S = 30.0
LINGER_MS = 20  # wait this long to fill a producer batch
BATCH_BYTES = 256 * 1024  # largest producer batch per partition


class KafkaUnavailableError(RuntimeError):
    """The broker cannot be reached."""

    def __init__(self, bootstrap: str) -> None:
        """Build the message ``cannot reach Kafka at <bootstrap>``.

        Args:
            bootstrap: The bootstrap servers that did not answer.
        """
        super().__init__(f"cannot reach Kafka at {bootstrap}")


class TopicNotFoundError(RuntimeError):
    """The topic does not exist on the broker."""

    def __init__(self, topic: str) -> None:
        """Build the message ``topic <topic> not found (run make produce first)``.

        Args:
            topic: The missing topic.
        """
        super().__init__(f"topic {topic} not found (run make produce first)")


class DeliveryError(RuntimeError):
    """A message could not be delivered to the broker."""


class FutureLike(Protocol):
    """The part of a Kafka send future that `publish_drive` uses."""

    def get(self, timeout: float | None = None) -> object:
        """Return the delivery result, or raise the delivery error."""
        ...


class ProducerLike(Protocol):
    """The part of ``kafka.KafkaProducer`` that `publish_drive` uses (a seam for fakes)."""

    def send(self, topic: str, *, key: bytes, value: bytes) -> object:
        """Queue one message.

        Args:
            topic: Topic name.
            key: Message key (the drive id as UTF-8 bytes).
            value: Message value (an encoded frame).

        Returns:
            Whatever the producer returns (a future for ``KafkaProducer``).
        """
        ...

    def flush(self) -> None:
        """Block until every queued message has been sent."""
        ...


def encode_frame(frame: dict[str, object]) -> bytes:
    """Encode a frame message as UTF-8 JSON with compact separators.

    Non-ASCII characters are written as they are (``ensure_ascii=False``). A frame whose
    ``points`` is not a list of triples (a malformed frame) is encoded unchanged: the
    function never validates content.

    Args:
        frame: The frame message.

    Returns:
        The encoded message, for example ``b'{"a":1}'``.
    """
    return json.dumps(frame, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def decode_frame(raw: bytes) -> dict[str, object]:
    """Decode a message produced by `encode_frame`.

    Args:
        raw: The message bytes.

    Returns:
        The frame message.

    Raises:
        ValueError: If ``raw`` is not valid UTF-8 JSON, or the JSON value is not an object.
    """
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"message is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"message is not a JSON object: {type(value).__name__}")
    return cast(dict[str, object], value)


def publish_drive(
    frames: list[dict[str, object]],
    bootstrap: str = BOOTSTRAP,
    topic: str = TOPIC,
    rate_fps: float | None = None,
    producer: ProducerLike | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Publish frame messages to Kafka in the given order.

    Each message has the key ``frame["drive_id"]`` encoded as UTF-8 bytes and the value
    ``encode_frame(frame)``, so one drive stays ordered in one partition. Frames are sent
    unchanged, including malformed ones. The producer is flushed once after the last message.

    Pacing: ``rate_fps=None`` sends as fast as possible and never calls ``sleep``. With a
    number, message ``i`` (counting from 0) is not sent before ``start + i / rate_fps``, where
    ``start`` is ``clock()`` read before the first message; the function waits with
    ``sleep(delay)`` for the missing time only (``delay > 0``, never a zero or negative sleep).

    Args:
        frames: Frame messages in emission order.
        bootstrap: Kafka bootstrap servers, used only when ``producer`` is ``None``.
        topic: Target topic.
        rate_fps: Frames per second, or ``None`` for no pacing.
        producer: A producer to use; when ``None`` `build_producer` makes one, closed before
            returning.
        sleep: Function that waits the given number of seconds (a seam for tests).
        clock: Monotonic clock in seconds (a seam for tests).

    Returns:
        The number of messages sent and acknowledged.

    Raises:
        KafkaUnavailableError: If the broker cannot be reached (no producer given).
        DeliveryError: If a message could not be delivered; nothing is reported as sent.
    """
    own = producer is None
    target = build_producer(bootstrap) if producer is None else producer
    try:
        return _send_all(frames, target, topic, rate_fps, sleep, clock)
    except KafkaError as exc:
        raise DeliveryError(f"delivery to Kafka failed: {exc}") from exc
    finally:
        if own:
            cast(KafkaProducer, target).close()


def _send_all(
    frames: list[dict[str, object]],
    producer: ProducerLike,
    topic: str,
    rate_fps: float | None,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> int:
    """Send the frames in order, paced if a rate is given, flush once, then check every send.

    A send that returned no future (a test fake) counts as delivered.

    Args:
        frames: Frame messages in emission order.
        producer: The producer to send with.
        topic: Target topic.
        rate_fps: Frames per second, or ``None`` for no pacing.
        sleep: Function that waits the given number of seconds.
        clock: Monotonic clock in seconds.

    Returns:
        The number of messages sent.
    """
    start = clock()
    futures: list[FutureLike] = []
    for i, frame in enumerate(frames):
        if rate_fps is not None:
            delay = start + i / rate_fps - clock()
            if delay > 0:
                sleep(delay)
        key = str(frame["drive_id"]).encode("utf-8")
        future = producer.send(topic, key=key, value=encode_frame(frame))
        if future is not None:
            futures.append(cast(FutureLike, future))
    producer.flush()
    for future in futures:
        future.get(timeout=_DELIVERY_TIMEOUT_S)  # raises the delivery error of a failed send
    return len(frames)


def build_producer(bootstrap: str) -> KafkaProducer:
    """Build a ``KafkaProducer`` that waits for all replicas and fails fast when unreachable.

    Args:
        bootstrap: Kafka bootstrap servers.

    Returns:
        The producer (``acks="all"``, 5 retries). Batches are compressed with gzip (no extra
        library needed; JSON of integers shrinks about 4x) and the producer waits up to
        ``LINGER_MS`` to fill a batch of up to ``BATCH_BYTES``, which trades a few milliseconds of
        latency for far fewer, larger requests.

    Raises:
        KafkaUnavailableError: If the broker does not answer within `CONNECT_TIMEOUT_MS`.
    """
    try:
        return KafkaProducer(
            bootstrap_servers=bootstrap,
            acks="all",
            retries=5,
            compression_type="gzip",
            linger_ms=LINGER_MS,
            batch_size=BATCH_BYTES,
            bootstrap_timeout_ms=CONNECT_TIMEOUT_MS,
            request_timeout_ms=CONNECT_TIMEOUT_MS,
            max_block_ms=CONNECT_TIMEOUT_MS,
        )
    except KafkaError as exc:
        raise KafkaUnavailableError(bootstrap) from exc


def check_broker(bootstrap: str, topic: str) -> None:
    """Fail fast unless the broker answers and has the topic.

    Args:
        bootstrap: Kafka bootstrap servers.
        topic: Topic that must exist.

    Raises:
        KafkaUnavailableError: If the broker does not answer within `CONNECT_TIMEOUT_MS`.
        TopicNotFoundError: If the broker has no such topic.
    """
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap,
            request_timeout_ms=CONNECT_TIMEOUT_MS,
            bootstrap_timeout_ms=CONNECT_TIMEOUT_MS,
        )
        try:
            topics = admin.list_topics()
        finally:
            admin.close()
    except KafkaError as exc:
        raise KafkaUnavailableError(bootstrap) from exc
    if topic not in topics:
        raise TopicNotFoundError(topic)
