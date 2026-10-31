"""Message v2 ("claim check"): the clouds go to a file per drive, Kafka carries a small message.

Role in the pipeline: the producer side of message v2 and the reader of its cloud references.
A v1 message carries the whole point cloud (about 15 MB of JSON at the assignment's 1M points);
a v2 message carries only what the pipeline uses (centroid in meters, point count) plus
``cloud_ref``, ``<cloud file>#<position>``, where the cloud file holds the drive's clouds encoded
by `vl.cloud_codec` and ``<position>`` is the frame's position in the drive's emission order. Here
the cloud file is a local path; in production it is an object-store key (``s3://...``) written
before the messages are sent.

A frame without a usable cloud (a malformed message) becomes a v2 message that breaks the v2
contract on purpose (``centroid_m`` is the string ``"malformed"``), so it is still rejected as
VL-005 downstream, like its v1 original.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vl.cloud_codec import Cloud, clouds_of, decode_clouds, encode_clouds
from vl.schemas import SCHEMA_V2

CLOUD_SUFFIX = ".vlc"  # "VL clouds": the codec container of one drive


def _v2_message(frame: Mapping[str, Any], cloud: Cloud | None, ref: str) -> dict[str, Any]:
    """Turn one v1 frame message into its v2 twin.

    Args:
        frame: The v1 message.
        cloud: Its cloud (``None`` when ``points`` is malformed).
        ref: Its ``cloud_ref``.

    Returns:
        The message without ``points``, with ``schema_version``, ``n_points``, ``centroid_m``
        (absent for an empty cloud, ``"malformed"`` for a malformed one) and ``cloud_ref``.
    """
    message = {key: value for key, value in frame.items() if key != "points"}
    message.update(schema_version=SCHEMA_V2, cloud_ref=ref)
    if cloud is None:
        return {**message, "n_points": 0, "centroid_m": "malformed"}
    message["n_points"] = len(cloud)
    if len(cloud):
        message["centroid_m"] = (cloud.mean(axis=0) / 100.0).tolist()
    return message


def to_v2(
    frames: Sequence[Mapping[str, Any]], cloud_dir: str, drive_id: str
) -> list[dict[str, Any]]:
    """Write a drive's clouds to its cloud file and return the v2 messages.

    Args:
        frames: The drive's v1 messages, in emission order.
        cloud_dir: Directory of the cloud files (created if missing).
        drive_id: The drive id, which names the file.

    Returns:
        One v2 message per frame, in the same order.
    """
    path = Path(cloud_dir) / f"{drive_id}{CLOUD_SUFFIX}"
    path.parent.mkdir(parents=True, exist_ok=True)
    clouds = clouds_of(frames)
    path.write_bytes(encode_clouds(clouds))
    refs = [f"{path}#{position}" for position in range(len(frames))]
    return [_v2_message(f, c, r) for f, c, r in zip(frames, clouds, refs, strict=True)]


def load_cloud(ref: str) -> Cloud | None:
    """Read the cloud a ``cloud_ref`` points to.

    Args:
        ref: ``<cloud file>#<position>``.

    Returns:
        The cloud in integer centimeters, exactly as produced; ``None`` for a malformed frame.

    Raises:
        ValueError: If ``ref`` has no ``#<position>``.
        OSError: If the cloud file cannot be read.
    """
    path, _, position = ref.rpartition("#")
    if not path or not position.isdigit():
        raise ValueError(f"cloud_ref must be '<file>#<position>', got {ref!r}")
    return decode_clouds(Path(path).read_bytes())[int(position)]
