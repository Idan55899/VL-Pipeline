"""Command line and file writer for the mock fleet.

Role in the pipeline: materializes simulated drives as ``frames.jsonl`` (the Kafka producer's
input) and ``labels.json`` (ground truth for checking the validator).

Usage: ``python -m vl.mock.generate --drives 5 --frames 1000 --seed 0 --out data/mock``
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from vl.mock.drive_sim import SimulatedDrive, simulate_fleet


def write_fleet_jsonl(drives: list[SimulatedDrive], out_dir: str) -> None:
    """Write ``frames.jsonl`` and ``labels.json`` for a fleet.

    ``frames.jsonl`` has one JSON message per line in emission order. ``labels.json`` is
    ``{"drives": [DriveMeta dicts], "labels": [FaultLabel dicts]}``.

    Args:
        drives: The simulated drives.
        out_dir: Output directory; created if missing.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(frame) for drive in drives for frame in drive.frames]
    (out / "frames.jsonl").write_text("".join(line + "\n" for line in lines))
    truth = {
        "drives": [asdict(drive.meta) for drive in drives],
        "labels": [asdict(label) for drive in drives for label in drive.labels],
    }
    (out / "labels.json").write_text(json.dumps(truth, indent=2) + "\n")


def at_least(minimum: int) -> Callable[[str], int]:
    """Build an argparse type that accepts integers of at least ``minimum``.

    Args:
        minimum: Smallest accepted value.

    Returns:
        A converter raising `argparse.ArgumentTypeError` for non-integers and small values.
    """

    def convert(text: str) -> int:
        """Convert one command-line value.

        Args:
            text: The raw argument.

        Returns:
            The integer value.

        Raises:
            argparse.ArgumentTypeError: If ``text`` is not an integer >= ``minimum``.
        """
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}, got {value}")
        return value

    return convert


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments; ``None`` means ``sys.argv[1:]``.

    Returns:
        The parsed ``drives``, ``frames``, ``seed`` and ``out`` options.
    """
    parser = argparse.ArgumentParser(description="Generate a mock fleet of drives.")
    parser.add_argument("--drives", type=at_least(1), default=5, help="number of drives")
    parser.add_argument("--frames", type=at_least(2), default=1000, help="frames per drive")
    parser.add_argument("--seed", type=at_least(0), default=0, help="fleet seed")
    parser.add_argument("--out", default="data/mock", help="output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the command line: simulate a fleet and write it to disk.

    Args:
        argv: Arguments (``--drives``, ``--frames``, ``--seed``, ``--out``); defaults to
            ``sys.argv[1:]``.
    """
    args = _parse_args(argv)
    drives = simulate_fleet(args.drives, seed=args.seed, n_frames=args.frames)
    write_fleet_jsonl(drives, args.out)


if __name__ == "__main__":
    main()
