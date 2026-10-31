"""Tests for the Parquet ``raw_frames`` source of the validator CLI (``vl.validate``).

``--source`` is the ``raw_frames`` table directory (hive partitions ``drive_id=...``); the
checks, ``--labels`` and the exit codes behave as for JSONL.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from pipeline_helpers import RAW_DTYPES, raw_from_frames
from vl.mock.drive_sim import SimConfig, SimulatedDrive, simulate_drive, simulate_fleet
from vl.mock.faults import Fault
from vl.mock.generate import write_fleet_jsonl
from vl.schemas import RAW_FRAMES_COLUMNS
from vl.validate import main
from vl.validation import CHECK_SOURCE_NOT_EMPTY

# Malformed lines can never reach a ``raw_frames`` table, so the fleets have none.
NO_MALFORMED = replace(SimConfig(), malformed_rate=0.0)
Run = tuple[int, str, str]


def run(args: list[str], capsys: pytest.CaptureFixture[str]) -> Run:
    """Run ``main`` and return the exit code, stdout and stderr."""
    code = main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def raw_table(drives: list[SimulatedDrive]) -> pd.DataFrame:
    """Build the ``raw_frames`` rows of a fleet, in emission order."""
    return pd.concat([raw_from_frames(drive.frames) for drive in drives], ignore_index=True)


def write_parquet_source(drives: list[SimulatedDrive], directory: Path) -> Path:
    """Write a fleet as a hive-partitioned ``raw_frames`` table and return its directory."""
    table = directory / "raw_frames"
    raw_table(drives).to_parquet(table, partition_cols=["drive_id"])
    return table


def write_jsonl_source(drives: list[SimulatedDrive], directory: Path) -> Path:
    """Write a fleet as ``frames.jsonl`` (and ``labels.json``) and return the frames file."""
    write_fleet_jsonl(drives, str(directory))
    return directory / "frames.jsonl"


def report_lines(out: str, skip: str = CHECK_SOURCE_NOT_EMPTY) -> list[str]:
    """Split the report into lines, dropping the check named ``skip``."""
    return [line for line in out.splitlines() if f" {skip}:" not in line]


def line_of(out: str, check: str) -> str:
    """Return the report line of one check."""
    return next(line for line in out.splitlines() if f" {check}:" in line)


# --- the same fleet through both sources ---------------------------------------------------


def test_validate_cli_parquet_source_exit_zero_and_matches_jsonl(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drives = simulate_fleet(3, seed=4, n_frames=400, sim=NO_MALFORMED)
    jsonl = write_jsonl_source(drives, tmp_path / "j")
    source = write_parquet_source(drives, tmp_path / "p")
    labels = ["--labels", str(tmp_path / "j" / "labels.json")]
    n_rows = sum(len(drive.frames) for drive in drives)
    for extra in ([], labels):  # without and with --labels: same report as for JSONL
        from_jsonl = run(["--source", str(jsonl), *extra], capsys)
        code, out, err = run(["--source", str(source), *extra], capsys)
        assert (code, err) == (0, ""), f"{extra}: {out}"
        assert from_jsonl[0] == 0, extra
        assert report_lines(out) == report_lines(from_jsonl[1]), extra
        row_line = f"PASS {CHECK_SOURCE_NOT_EMPTY}: {n_rows} row(s)"
        assert line_of(out, CHECK_SOURCE_NOT_EMPTY) == row_line, extra
    assert "labeled_faults_detected" in out


def test_validate_cli_parquet_numeric_looking_drive_ids_stay_strings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    faults = (Fault("drop", 100),)
    for number, ids in enumerate((("007", "12"), ("12", "13"))):
        drives = [
            simulate_drive(drive_id, n_frames=300, sim=SimConfig.clean(), faults=faults)
            for drive_id in ids
        ]
        write_fleet_jsonl(drives, str(tmp_path / f"j{number}"))
        source = write_parquet_source(drives, tmp_path / f"p{number}")
        labels = ["--labels", str(tmp_path / f"j{number}" / "labels.json")]
        code, out, _ = run(["--source", str(source), *labels], capsys)
        assert code == 0, f"{ids}: {out}"
        # the string ids match the labels
        assert "PASS labeled_faults_detected: 2/2 detected" in out, ids


# --- no rows -------------------------------------------------------------------------------


def empty_zero_row_file(table: Path) -> None:
    """Create a ``raw_frames`` table that holds one Parquet file with the schema and no rows."""
    table.mkdir(parents=True)
    empty = pd.DataFrame(columns=RAW_FRAMES_COLUMNS).astype(RAW_DTYPES)
    empty.to_parquet(table / "part-0.parquet")


def empty_partition_directory(table: Path) -> None:
    """Create a ``raw_frames`` table that holds a ``drive_id=...`` directory without data."""
    (table / "drive_id=d1").mkdir(parents=True)


def test_validate_cli_empty_parquet_directory_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    makers: list[Callable[[Path], None]] = [empty_zero_row_file, empty_partition_directory]
    for make_empty in makers:
        table = tmp_path / make_empty.__name__ / "raw_frames"
        make_empty(table)
        code, out, err = run(["--source", str(table)], capsys)
        assert code == 1, f"{make_empty.__name__}: {out}"
        assert err == "", make_empty.__name__
        expected = f"FAIL {CHECK_SOURCE_NOT_EMPTY}: 0 row(s)"
        assert line_of(out, CHECK_SOURCE_NOT_EMPTY) == expected, make_empty.__name__


def test_validate_cli_bare_empty_directory_is_an_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "bare").mkdir()
    code, out, err = run(["--source", str(tmp_path / "bare")], capsys)
    assert code == 2
    assert out == ""
    assert len(err.strip().splitlines()) == 1
