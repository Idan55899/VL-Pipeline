"""Command line producer: simulate a fleet and publish its frames to Kafka.

Role in the pipeline: the live data source of the demo. It builds the simulated drives, sends
every message through `vl.kafka_io.publish_drive` in emission order, and can also write the
fleet to disk (``frames.jsonl`` and ``labels.json``) for the validator.

Usage: ``python -m vl.mock.producer --drives 3 --frames 1000 --seed 0 --faults demo
[--rate-fps 16] [--bootstrap localhost:9092] [--topic vl.frames] [--out-dir data/mock]``
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace

from vl.claim_check import to_v2
from vl.kafka_io import (
    BOOTSTRAP,
    TOPIC,
    DeliveryError,
    KafkaUnavailableError,
    ProducerLike,
    publish_drive,
)
from vl.mock.drive_sim import SimConfig, SimulatedDrive, simulate_drive, simulate_fleet
from vl.mock.faults import Fault
from vl.mock.generate import at_least, write_fleet_jsonl

_NO_RANDOM_FAULTS = replace(
    SimConfig(),
    drop_burst_rate=0.0,
    duplicate_rate=0.0,
    sparse_rate=0.0,
    jump_rate=0.0,
    malformed_rate=0.0,
    late_arrival_rate=0.0,
)
_DEMO_KINDS = ("drop", "jump", "duplicate", "malformed")


def _demo_faults(frames: int) -> tuple[Fault, ...]:
    """Place the four demo faults at 20%, 40%, 60% and 80% of a drive.

    Args:
        frames: Nominal frames per drive.

    Returns:
        One ``drop``, ``jump``, ``duplicate`` and ``malformed`` fault, in that order.
    """
    return tuple(Fault(kind, (i + 1) * frames // 5) for i, kind in enumerate(_DEMO_KINDS))


def build_fleet(
    drives: int, frames: int, seed: int, faults: str, scripted: bool = False
) -> list[SimulatedDrive]:
    """Simulate the fleet of the producer.

    The drives, profiles, vehicle kinds and seeds are those of `simulate_fleet`
    (``drive-001``, ``drive-002``, ...; the call is `simulate_fleet` with its explicit-faults
    parameter), with the default sensor noise and timestamp jitter but every random fault rate
    set to zero, so the faults of the fleet are exactly the explicit ones. The same arguments
    give an identical fleet.

    ``faults="none"`` adds no fault. ``faults="demo"`` adds, to ``drive-002`` only, one
    ``drop``, one ``jump``, one ``duplicate`` and one ``malformed`` fault at four distinct
    frames, at least 10 frames apart and inside the drive (computed from ``frames``, for
    example at 20%, 40%, 60% and 80% of it). With fewer than two drives, ``demo`` adds nothing.

    ``scripted=True`` appends one more drive, ``drive-<drives + 1>``, on the ``scripted`` profile
    (a car, no faults): its fixed scenario yields one HARD_BRAKE, one SHARP_TURN and one STOP, so
    the demo shows every event type of the use case (the random profiles stay below the
    hard-brake and hard-accel thresholds). The scenario needs about 830 frames.

    Args:
        drives: Number of drives (at least 1).
        frames: Nominal frames per drive (at least 50).
        seed: Fleet seed.
        faults: ``"none"`` or ``"demo"``.
        scripted: Whether to append the scripted scenario drive.

    Returns:
        The simulated drives in id order.

    Raises:
        ValueError: If ``faults`` is neither ``"none"`` nor ``"demo"``.
    """
    if faults not in ("none", "demo"):
        raise ValueError(f"faults must be 'none' or 'demo', got {faults!r}")
    explicit = {1: _demo_faults(frames)} if faults == "demo" and drives >= 2 else None
    fleet = simulate_fleet(drives, seed, frames, _NO_RANDOM_FAULTS, explicit)
    if scripted:
        drive_id = f"drive-{drives + 1:03d}"
        fleet.append(simulate_drive(drive_id, "scripted", "car", frames, seed, _NO_RANDOM_FAULTS))
    return fleet


def _as_v2(drive: SimulatedDrive, cloud_dir: str) -> SimulatedDrive:
    """Replace a drive's messages with their v2 twins, writing its cloud file.

    Args:
        drive: The simulated drive.
        cloud_dir: Directory of the cloud files.

    Returns:
        The same drive with v2 messages (labels and metadata unchanged).
    """
    return replace(drive, frames=to_v2(drive.frames, cloud_dir, drive.meta.drive_id))


def _positive_float(text: str) -> float:
    """Convert a command-line value to a float greater than zero.

    Args:
        text: The raw argument.

    Returns:
        The value.

    Raises:
        argparse.ArgumentTypeError: If ``text`` is not a number greater than zero.
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the producer command line.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The namespace with drives, frames, seed, faults, rate_fps, bootstrap, topic, out_dir.
    """
    parser = argparse.ArgumentParser(prog="vl.mock.producer", description=__doc__)
    parser.add_argument("--drives", type=at_least(1), default=3)
    parser.add_argument("--frames", type=at_least(50), default=1000)
    parser.add_argument("--seed", type=at_least(0), default=0)
    parser.add_argument("--faults", choices=["none", "demo"], default="none")
    parser.add_argument("--scripted", action="store_true", help="append the scripted drive")
    parser.add_argument("--rate-fps", type=_positive_float, default=None)
    parser.add_argument("--bootstrap", default=BOOTSTRAP)
    parser.add_argument("--topic", default=TOPIC)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cloud-dir", default=None, help="send v2 messages, clouds stored here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, producer: ProducerLike | None = None) -> int:
    """Run the producer command line.

    Builds the fleet with `build_fleet`, publishes every message of every drive through
    one `publish_drive` call (drive after drive, each in emission order, one producer and one
    pacing clock for the whole fleet; ``--rate-fps`` is passed through, ``--bootstrap`` and
    ``--topic`` default to ``BOOTSTRAP`` and ``TOPIC``). With
    ``--out-dir`` it also writes ``frames.jsonl`` and ``labels.json`` there (`write_fleet_jsonl`).
    With ``--cloud-dir`` every drive's clouds are written there (`vl.claim_check.to_v2`) and the
    messages sent and written are the small v2 messages instead of the v1 ones.
    Prints one summary line with the number of messages sent and the number of drives.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.
        producer: A producer passed to `publish_drive` (a seam for tests); ``None`` builds a
            real ``KafkaProducer``.

    Returns:
        ``0`` on success; ``1`` (one ``error: ...`` line on stderr) when the broker cannot be
        reached or a message is not delivered. Bad arguments exit with status ``2`` (argparse).
    """
    args = _parse_args(argv)
    fleet = build_fleet(args.drives, args.frames, args.seed, args.faults, args.scripted)
    if args.cloud_dir is not None:
        fleet = [_as_v2(drive, args.cloud_dir) for drive in fleet]
    messages = [frame for drive in fleet for frame in drive.frames]
    try:
        sent = publish_drive(messages, args.bootstrap, args.topic, args.rate_fps, producer=producer)
    except (KafkaUnavailableError, DeliveryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.out_dir is not None:
        write_fleet_jsonl(fleet, args.out_dir)
    print(f"sent {sent} messages from {len(fleet)} drives to {args.topic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
