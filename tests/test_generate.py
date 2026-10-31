"""Tests for writing a simulated fleet to ``frames.jsonl`` and ``labels.json``."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from sim_helpers import Frame
from vl.mock.drive_sim import SimConfig, SimulatedDrive, simulate_fleet
from vl.mock.generate import main, write_fleet_jsonl

FAULTY = replace(
    SimConfig(), drop_burst_rate=0.02, duplicate_rate=0.01, sparse_rate=0.01, malformed_rate=0.01
)


def read_frames(path: Path) -> list[Frame]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def by_drive(frames: list[Frame]) -> dict[str, list[Frame]]:
    grouped: dict[str, list[Frame]] = {}
    for frame in frames:
        grouped.setdefault(frame["drive_id"], []).append(frame)
    return grouped


def label_key(label: dict[str, Any]) -> tuple[str, int, str]:
    return (label["drive_id"], label["frame_idx"], label["kind"])


def check_labels(drives: list[SimulatedDrive], path: Path) -> None:
    data = json.loads(path.read_text())
    assert data["drives"] == [asdict(drive.meta) for drive in drives]
    expected = [asdict(label) for drive in drives for label in drive.labels]
    assert expected and sorted(data["labels"], key=label_key) == sorted(expected, key=label_key)


def test_write_fleet_jsonl_roundtrip(tmp_path: Path) -> None:
    drives = simulate_fleet(3, seed=1, n_frames=200, sim=FAULTY)
    write_fleet_jsonl(drives, str(tmp_path))
    frames = read_frames(tmp_path / "frames.jsonl")
    assert len(frames) == sum(len(drive.frames) for drive in drives)
    grouped = by_drive(frames)
    assert all(grouped[drive.meta.drive_id] == drive.frames for drive in drives)
    check_labels(drives, tmp_path / "labels.json")
    labels = json.loads((tmp_path / "labels.json").read_text())["labels"]
    assert {"malformed", "gap_after_malformed"} <= {label["kind"] for label in labels}


def test_cli_rejects_bad_arguments_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    for args in [["--drives", "0"], ["--frames", "1"], ["--seed", "-1"], ["--drives", "x"]]:
        with pytest.raises(SystemExit) as caught:
            main(args)
        assert caught.value.code == 2, args
        assert args[0] in capsys.readouterr().err, args


def test_cli_writes_files_for_minimal_valid_arguments(tmp_path: Path) -> None:
    main(["--drives", "1", "--frames", "2", "--seed", "0", "--out", str(tmp_path)])
    assert len(read_frames(tmp_path / "frames.jsonl")) == 2
