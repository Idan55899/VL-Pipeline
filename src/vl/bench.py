"""Benchmark of the pipeline's hot spots, printed as a Markdown table.

Role in the pipeline: evidence for the scaling design. It measures, in memory and without
Spark, Kafka or disk, what one frame costs on the wire and to decode as JSON (message v1)
against a binary int16 cloud, how small the rigid-body cloud codec (`vl.cloud_codec`) makes a
drive's clouds, and how fast ``derive_frames`` runs per drive.

Usage: ``python -m vl.bench`` (full sizes up to the PDF's 1M points per frame, about 20 s) or
``python -m vl.bench --quick`` (small sizes, about 1 s).
"""

import argparse
import json
import time
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from vl.cloud_codec import clouds_of, decode_clouds, encode_clouds
from vl.etl.features import derive_frames
from vl.mock.drive_sim import SimConfig, simulate_drive
from vl.schemas import RAW_FRAMES_COLUMNS

QUICK_POINTS = (200, 10_000)
FULL_POINTS = (200, 100_000, 1_000_000)  # 1M points: the resolution of the assignment
QUICK_DRIVES = (1000,)
FULL_DRIVES = (1000, 57_600)  # 57,600 frames: one hour at 16 fps
QUICK_CODEC_FRAMES, FULL_CODEC_FRAMES = 200, 1000


class BenchRow(NamedTuple):
    """One measurement.

    Attributes:
        name: What was measured, including the size it was measured at.
        value: The measured number.
        unit: The unit of ``value``.
    """

    name: str
    value: float
    unit: str


def _seconds(action: Callable[[], object], repeat: int = 3) -> float:
    """Time an action, keeping the fastest of ``repeat`` runs.

    Args:
        action: The function to time.
        repeat: Number of runs.

    Returns:
        The fastest run in seconds.
    """
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        action()
        best = min(best, time.perf_counter() - start)
    return best


def _cloud(points: int) -> npt.NDArray[np.int16]:
    """Build a seeded random cloud of integer centimeters.

    Args:
        points: Number of points.

    Returns:
        An ``(points, 3)`` int16 array.
    """
    return np.random.default_rng(0).integers(-500, 500, size=(points, 3)).astype(np.int16)


def _wire_rows(points: int) -> list[BenchRow]:
    """Measure one frame's size and decode-plus-centroid time as JSON and as binary.

    Args:
        points: Points in the frame.

    Returns:
        Four rows: the size and decode time of each format.
    """
    cloud = _cloud(points)
    text = json.dumps(cloud.tolist(), separators=(",", ":")).encode()
    blob = cloud.tobytes()
    json_s = _seconds(lambda: np.asarray(json.loads(text), dtype=np.float64).mean(axis=0))
    binary_s = _seconds(lambda: np.frombuffer(blob, np.int16).reshape(-1, 3).mean(axis=0))
    at = f"@{points} pts"
    return [
        BenchRow(f"json bytes/frame {at}", len(text), "B"),
        BenchRow(f"binary bytes/frame {at}", len(blob), "B"),
        BenchRow(f"json decode+centroid {at}", json_s * 1000, "ms"),
        BenchRow(f"binary decode+centroid {at}", binary_s * 1000, "ms"),
    ]


def _raw_drive(n_frames: int) -> pd.DataFrame:
    """Build the raw rows of one straight drive at 15 m/s.

    Args:
        n_frames: Frames in the drive.

    Returns:
        A ``RAW_FRAMES_COLUMNS`` DataFrame.
    """
    frame_idx = np.arange(n_frames)
    ts = frame_idx * 0.0625
    raw = pd.DataFrame({"frame_idx": frame_idx, "ts": ts, "x": ts * 15.0})
    raw = raw.assign(drive_id="d", vehicle_id="v", vehicle_kind="car", n_points=200, y=0.0, z=0.0)
    return raw.assign(drive_start_utc="2026-01-01T00:00:00+00:00")[RAW_FRAMES_COLUMNS]


def _transform_row(n_frames: int) -> BenchRow:
    """Measure ``derive_frames`` throughput on one drive.

    Args:
        n_frames: Frames in the drive.

    Returns:
        Rows per second.
    """
    raw = _raw_drive(n_frames)
    seconds = _seconds(lambda: derive_frames(raw))
    return BenchRow(f"derive_frames rows/s @{n_frames}-frame drive", n_frames / seconds, "rows/s")


def _codec_rows(n_frames: int) -> list[BenchRow]:
    """Measure the rigid-body codec on one simulated city drive (noise and jitter on).

    Args:
        n_frames: Frames in the drive.

    Returns:
        The JSON size of the drive's clouds, the codec's size, and encode and decode times.
    """
    clouds = clouds_of(simulate_drive("d", "city", "car", n_frames, 0, SimConfig()).frames)
    text = sum(len(json.dumps(c.tolist(), separators=(",", ":"))) for c in clouds if c is not None)
    blob = encode_clouds(clouds)
    at = f"@{n_frames}-frame drive"
    return [
        BenchRow(f"json cloud bytes/drive {at}", text, "B"),
        BenchRow(f"codec bytes/drive {at}", len(blob), "B"),
        BenchRow(f"codec encode {at}", _seconds(lambda: encode_clouds(clouds)) * 1000, "ms"),
        BenchRow(f"codec decode {at}", _seconds(lambda: decode_clouds(blob)) * 1000, "ms"),
    ]


def run_bench(quick: bool = False) -> list[BenchRow]:
    """Run every measurement.

    Args:
        quick: Use the small sizes (for tests and a fast look).

    Returns:
        The measurements, in a stable order.
    """
    points = QUICK_POINTS if quick else FULL_POINTS
    drives = QUICK_DRIVES if quick else FULL_DRIVES
    rows = [row for n in points for row in _wire_rows(n)]
    rows += _codec_rows(QUICK_CODEC_FRAMES if quick else FULL_CODEC_FRAMES)
    return rows + [_transform_row(n) for n in drives]


def to_markdown(rows: Sequence[BenchRow]) -> str:
    """Render measurements as a Markdown table.

    Args:
        rows: The measurements.

    Returns:
        The table text, starting with the header line.
    """
    lines = ["| measurement | value | unit |", "|---|---:|---|"]
    lines += [f"| {r.name} | {r.value:,.2f} | {r.unit} |" for r in rows]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and print the table.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        ``0``.
    """
    parser = argparse.ArgumentParser(description="Benchmark the pipeline's hot spots.")
    parser.add_argument("--quick", action="store_true", help="small sizes, about 1 s")
    print(to_markdown(run_bench(parser.parse_args(argv).quick)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
