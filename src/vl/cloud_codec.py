"""Lossless rigid-body codec for the point clouds of one drive.

Role in the pipeline: the storage format of message v2's raw clouds (the "claim check": Kafka
carries a small message with the centroid and a reference, the clouds live in a file per drive).

Why it compresses: the VL cloud is the vehicle's own body, so every frame is the same shape moved
somewhere else. The codec stores the shape once, then per frame only the integer shift (3 numbers)
and the residual ``cloud - shape - shift``, which sensor noise keeps within a few centimeters
(``int8``). A frame that does not fit the shape (another point count, or a residual beyond the
``int8`` range: a deformed or corrupt cloud) is stored raw. Arithmetic is ``int64`` throughout, so
decoding returns exactly the input (any 64-bit coordinates). ``None`` (a frame with no usable cloud,
such as a malformed message) round-trips as ``None``. The point order of a frame must match the
shape's for the frame to be stored as rigid; otherwise it is simply stored raw.

The container is NumPy's compressed ``.npz`` (zip + deflate), read without pickle.
"""

import io
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

Cloud = npt.NDArray[np.int64]
MISSING, RIGID, RAW = 0, 1, 2  # per-frame storage kind
_INT8 = np.iinfo(np.int8)


def clouds_of(frames: Sequence[Mapping[str, object]]) -> list[Cloud | None]:
    """Extract the clouds of frame messages, in order.

    Args:
        frames: Frame messages (``points`` is a list of ``[x, y, z]`` integer centimeters).

    Returns:
        One ``(n, 3)`` int64 array per message; ``None`` where ``points`` is not a list of integer
        triples (a malformed message).
    """
    return [_as_cloud(frame.get("points")) for frame in frames]


def _as_cloud(points: object) -> Cloud | None:
    """Convert one ``points`` value to a cloud array.

    Args:
        points: The message's ``points`` value.

    Returns:
        An ``(n, 3)`` int64 array, or ``None`` when ``points`` is not a list of integer triples.
    """
    if not isinstance(points, list):
        return None
    if not all(isinstance(p, list) and len(p) == 3 for p in points):
        return None
    if not all(type(v) is int for p in points for v in p):
        return None
    return np.array(points, dtype=np.int64).reshape(-1, 3)


def _reference(clouds: Sequence[Cloud | None]) -> Cloud:
    """Pick the shape: the first cloud with the most common point count, centered on the origin.

    Args:
        clouds: The clouds of the drive.

    Returns:
        The shape (``(n, 3)`` int64); empty when no frame has a cloud.
    """
    counts = Counter(len(c) for c in clouds if c is not None)
    if not counts:
        return np.zeros((0, 3), dtype=np.int64)
    size = counts.most_common(1)[0][0]
    first = next(c for c in clouds if c is not None and len(c) == size)
    return first - np.rint(first.mean(axis=0)).astype(np.int64) if size else first


def _fit(cloud: Cloud, shape: Cloud) -> tuple[Cloud, npt.NDArray[np.int8]] | None:
    """Express a cloud as ``shape + shift + residual`` when it fits.

    Args:
        cloud: The frame's cloud.
        shape: The drive's shape.

    Returns:
        ``(shift, residual)`` with an ``int8`` residual, or ``None`` when the point count differs,
        the cloud is empty, or a residual is outside the ``int8`` range.
    """
    if len(cloud) != len(shape) or len(cloud) == 0:
        return None
    shift = np.rint((cloud - shape).mean(axis=0)).astype(np.int64)
    residual = cloud - shape - shift
    if residual.min() < _INT8.min or residual.max() > _INT8.max:
        return None
    return shift, residual.astype(np.int8)


def _kind(cloud: Cloud | None, fitted: object) -> int:
    """Name the storage kind of one frame.

    Args:
        cloud: The frame's cloud.
        fitted: The result of `_fit` for it.

    Returns:
        ``MISSING``, ``RIGID`` or ``RAW``.
    """
    if cloud is None:
        return MISSING
    return RIGID if fitted is not None else RAW


def _arrays(clouds: Sequence[Cloud | None], shape: Cloud) -> dict[str, npt.NDArray[np.generic]]:
    """Split the clouds of a drive into the container's arrays.

    Args:
        clouds: The clouds in emission order.
        shape: The drive's shape.

    Returns:
        ``kinds``, ``shape``, ``shifts``, ``residuals``, ``raw`` and ``raw_counts``.
    """
    fits = [None if c is None else _fit(c, shape) for c in clouds]
    kinds = [_kind(c, f) for c, f in zip(clouds, fits, strict=True)]
    rigid = [f for f in fits if f is not None]
    raws = [c for c, k in zip(clouds, kinds, strict=True) if c is not None and k == RAW]
    return {
        "kinds": np.array(kinds, dtype=np.int8),
        "shape": shape,
        "shifts": np.array([f[0] for f in rigid], dtype=np.int64).reshape(-1, 3),
        "residuals": np.array([f[1] for f in rigid], dtype=np.int8).reshape(-1, len(shape), 3),
        "raw": np.concatenate(raws) if raws else np.zeros((0, 3), dtype=np.int64),
        "raw_counts": np.array([len(r) for r in raws], dtype=np.int64),
    }


def encode_clouds(clouds: Sequence[Cloud | None]) -> bytes:
    """Encode the clouds of one drive.

    Args:
        clouds: The clouds in emission order (``None`` for a frame without a usable cloud).

    Returns:
        The compressed container bytes.
    """
    buffer = io.BytesIO()
    # the stub types every keyword as the ``allow_pickle`` flag; the names here are array names
    np.savez_compressed(buffer, **_arrays(clouds, _reference(clouds)))  # type: ignore[arg-type]
    return buffer.getvalue()


def decode_clouds(blob: bytes) -> list[Cloud | None]:
    """Decode what `encode_clouds` wrote.

    Args:
        blob: The container bytes.

    Returns:
        The clouds, exactly as encoded, in the same order.
    """
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    rigid = iter(arrays["shape"] + arrays["shifts"][:, None, :] + arrays["residuals"])
    raw = iter(np.split(arrays["raw"], np.cumsum(arrays["raw_counts"])[:-1]))
    sources = {RIGID: rigid, RAW: raw}
    return [None if kind == MISSING else next(sources[int(kind)]) for kind in arrays["kinds"]]


def max_residuals(clouds: Sequence[Cloud | None]) -> list[int | None]:
    """Report how far each frame is from the rigid shape (a deformation or corruption signal).

    Args:
        clouds: The clouds of one drive.

    Returns:
        Per frame, the largest absolute residual in centimeters; ``None`` for a frame that does not
        fit the shape (stored raw) or has no cloud.
    """
    shape = _reference(clouds)
    fits = [None if c is None else _fit(c, shape) for c in clouds]
    return [None if f is None else int(np.abs(f[1]).max()) for f in fits]
