"""Command line validator: derive frames from a mock fleet and print the data-quality report.

Role in the pipeline: the standalone check of the mock data (and of the simulator-versus-
validator agreement); it uses the same rule functions as the ETL through ``derive_frames``.

The source is a ``frames.jsonl`` file or a Parquet ``raw_frames`` table directory (the
warehouse table written by the ingest, hive partitions ``drive_id=...``).

With ``--labels`` (the generator's ``labels.json``) the injected faults are checked too. With
``--realism`` (which needs ``--labels``) each drive is also checked against the physical limits
of its profile and vehicle kind, both taken from the ``drives`` metadata of ``labels.json``.
These are statistical limits (maximum speed, 99th percentile of acceleration and yaw rate): a
healthy drive near a limit can fail (seen on 1 of 40 default fleets, a parking drive with
``accel_p99`` 2.003 against 2.0) and the estimates are noisy on short drives (about 300
frames), so use at least 1000 frames per drive.

Usage: ``python -m vl.validate --source data/mock/frames.jsonl [--labels data/mock/labels.json
[--realism]]`` or ``python -m vl.validate --source data/warehouse/raw_frames``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import replace
from itertools import chain
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from vl.config import Config, load_config
from vl.errors import ErrorCode
from vl.etl.features import METADATA_COLUMNS, derive_frames
from vl.schemas import (
    INT64_MAX,
    INT64_MIN,
    MAX_POINTS,
    MESSAGE_STRING_FIELDS,
    RAW_FRAMES_COLUMNS,
    RAW_ORDER_COLUMNS,
    SCHEMA_V2,
)
from vl.validation import (
    CHECK_CODE_COUNTS,
    CHECK_FALSE_POSITIVES,
    CHECK_FLAG_RATE,
    CHECK_LABELED_FAULTS,
    CHECK_SOURCE_NOT_EMPTY,
    CheckResult,
    ValidationReport,
    check_error_code_counts,
    check_flag_rate,
    validate_frames,
    validate_realism,
)

MALFORMED_CODE = ErrorCode.MALFORMED_MESSAGE.value
CHECK_DUPLICATES = "duplicates_collapsed"  # informational: always passes
CHECK_FRAMES_LOST = "frames_lost"  # informational: always passes
EXIT_FAILED = 1  # a check failed
# the input cannot be used: unreadable source or labels, --realism without --labels, a source
# drive missing from the labels ``drives``, or an invalid VL_* environment value
EXIT_BAD_INPUT = 2
LABEL_COLUMNS = ["drive_id", "frame_idx", "kind", "expected_code"]
KEY = ["drive_id", "frame_idx"]
MAX_LISTED_MISSES = 5
REALISM_NEEDS_LABELS = (
    "--realism needs --labels (the profile and vehicle kind come from labels.json)"
)
# Hive partitioning of a ``raw_frames`` table with ``drive_id`` typed as a string: plain
# pyarrow discovery would infer an integer for ids such as "12" and lose the zeros of "007".
PARQUET_PARTITIONING = ds.partitioning(pa.schema([("drive_id", pa.string())]), flavor="hive")


class _InputError(Exception):
    """The input cannot be used (exit code 2).

    Raised for an unreadable or malformed source or labels file, ``--realism`` without
    ``--labels``, a source drive missing from the labels ``drives`` metadata, and an invalid
    ``VL_*`` environment value.
    """


# --- reading the frames ---------------------------------------------------------------------


def _is_number(value: object) -> bool:
    """Tell whether ``value`` is a finite int or float (a bool is not a number here).

    Args:
        value: Any JSON value.

    Returns:
        ``True`` for a finite int or float.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int64(value: object) -> bool:
    """Tell whether ``value`` is a JSON integer in the int64 range (a bool or float is not).

    Args:
        value: Any JSON value.

    Returns:
        ``True`` for an ``int`` that fits ``long``, as the Spark ingest requires.
    """
    return type(value) is int and INT64_MIN <= value <= INT64_MAX


def _centroid(points: object) -> tuple[np.ndarray[Any, Any], int] | None:
    """Compute the centroid of a point cloud in meters (clouds are in integer centimeters).

    Checks the shape first (a list of 3-element lists), then the coordinate types in one pass at C
    speed (``map(type, ...)``: every coordinate must be exactly ``int``, so a bool or a float is
    rejected), and lets NumPy reject an integer beyond int64 while building the array. This is
    about 2.4x faster than checking every point in Python (0.6 s against 1.5 s at 1M points).

    Args:
        points: The ``points`` field of a message.

    Returns:
        ``(centroid_m, n_points)``; an empty list is a valid empty cloud with a NaN centroid
        and 0 points. ``None`` when ``points`` is not a list of ``[x, y, z]`` int64 triples.
    """
    if type(points) is not list or not all(type(p) is list and len(p) == 3 for p in points):
        return None
    if set(map(type, chain.from_iterable(points))) - {int}:
        return None
    if not points:
        return np.full(3, np.nan), 0
    try:
        cloud = np.array(points, dtype=np.int64)
    except OverflowError:  # an integer beyond int64
        return None
    return cloud.mean(axis=0, dtype=np.float64) / 100.0, len(points)


def _valid_strings(message: dict[str, Any]) -> bool:
    """Check the string fields: ``drive_id`` a string, the others a string, null or absent.

    Args:
        message: A decoded JSON object.

    Returns:
        ``True`` when the message satisfies the contract on its string fields.
    """
    optional = (message.get(name) for name in MESSAGE_STRING_FIELDS[1:])
    return isinstance(message.get("drive_id"), str) and all(
        v is None or isinstance(v, str) for v in optional
    )


def _version(message: dict[str, Any]) -> int | None:
    """Read the message version.

    Args:
        message: A decoded JSON object.

    Returns:
        ``1`` when ``schema_version`` is absent, null or the integer 1; ``SCHEMA_V2`` for the
        integer 2; ``None`` for anything else (a string, a float, a bool, another number).
    """
    version = message.get("schema_version")
    if version is None:
        return 1
    return version if type(version) is int and version in (1, SCHEMA_V2) else None


def _cloud_v2(message: dict[str, Any]) -> tuple[np.ndarray[Any, Any], int] | None:
    """Read the centroid and point count of a v2 message.

    Args:
        message: A decoded v2 JSON object.

    Returns:
        ``(centroid_m, n_points)`` when ``cloud_ref`` is a string, ``n_points`` an integer from 0 to
        ``MAX_POINTS`` and
        ``centroid_m`` three finite numbers (absent or null when ``n_points`` is 0, which gives a
        NaN centroid); ``None`` otherwise.
    """
    n_points, centroid = message.get("n_points"), message.get("centroid_m")
    if not isinstance(message.get("cloud_ref"), str) or type(n_points) is not int:
        return None
    if not 0 <= n_points <= MAX_POINTS:
        return None
    if n_points == 0:
        return (np.full(3, np.nan), 0) if centroid is None else None
    if not isinstance(centroid, list) or len(centroid) != 3:
        return None
    if not all(_is_number(value) for value in centroid):
        return None
    return np.asarray(centroid, dtype=np.float64), n_points


def _cloud(message: dict[str, Any]) -> tuple[np.ndarray[Any, Any], int] | None:
    """Read the centroid and point count of a v1 or v2 message.

    Args:
        message: A decoded JSON object.

    Returns:
        ``(centroid_m, n_points)``, or ``None`` when the version or the cloud fields are invalid.
    """
    version = _version(message)
    if version == SCHEMA_V2:
        return _cloud_v2(message)
    return _centroid(message.get("points")) if version == 1 else None


def _raw_row(message: object) -> dict[str, Any] | None:
    """Convert one parsed message into a raw-frames row (the Spark ingest's strict contract).

    Args:
        message: A decoded JSON value.

    Returns:
        The row (``RAW_FRAMES_COLUMNS`` keys), or ``None`` when the value is not a valid frame
        message: string fields strings, ``frame_idx`` an int64, ``ts`` a finite number, and
        either (v1) ``points`` a list of ``[x, y, z]`` integer triples (possibly empty, which is
        VL-006) or (v2, ``schema_version`` 2) the fields `_cloud_v2` checks.
    """
    if not isinstance(message, dict) or not _valid_strings(message):
        return None
    frame_idx, ts = message.get("frame_idx"), message.get("ts")
    cloud = _cloud(message)
    if cloud is None or not _is_number(ts) or not _is_int64(frame_idx):
        return None
    (x, y, z), n_points = cloud
    row = {name: message.get(name) for name in RAW_FRAMES_COLUMNS[:4]}
    return {**row, "frame_idx": frame_idx, "ts": ts, "n_points": n_points, "x": x, "y": y, "z": z}


def _reject_constant(name: str) -> NoReturn:
    """Reject the non-JSON constants ``NaN`` and ``Infinity`` while parsing.

    Args:
        name: The constant literal found by the parser.

    Raises:
        ValueError: Always.
    """
    raise ValueError(name)


def _parse_line(line: str) -> dict[str, Any] | None:
    """Parse one JSON line into a raw-frames row.

    Args:
        line: One line of ``frames.jsonl``.

    Returns:
        The row, or ``None`` when the line is blank, not JSON or not a valid frame message.
        Lines the parser cannot handle count as invalid: nesting beyond the recursion limit
        (``RecursionError``), integers too large for a float (``OverflowError``) and the ``NaN`` /
        ``Infinity`` tokens anywhere in the message.
    """
    try:
        return _raw_row(json.loads(line, parse_constant=_reject_constant))
    # Nesting deeper than the recursion limit (about 1000 levels) is malformed even inside an
    # ignored extra field: a documented limitation, accepted.
    except (ValueError, RecursionError, OverflowError):
        return None


class _Source(NamedTuple):
    """The parsed content of a source: a ``frames.jsonl`` file or a Parquet ``raw_frames`` table.

    Attributes:
        raw: Raw rows of the valid messages (file order for JSONL, partition order for
            Parquet).
        n_lines: Number of lines (JSONL) or rows (Parquet).
        n_malformed: Number of lines that are not valid frame messages (always 0 for
            Parquet: a table holds only valid frames).
        unit: What ``n_lines`` counts, ``"line"`` or ``"row"``, for the ``source_not_empty``
            detail.
    """

    raw: pd.DataFrame
    n_lines: int
    n_malformed: int
    unit: str = "line"


def _one_line(exc: Exception) -> str:
    """Render an exception as a single line.

    Args:
        exc: The exception.

    Returns:
        Its message with line breaks collapsed (the class name when the message is empty).
    """
    return " ".join(str(exc).split()) or type(exc).__name__


def _read_text(path: Path, what: str) -> str:
    """Read a UTF-8 text file.

    Args:
        path: The file.
        what: What the file is, for the error message (``source`` or ``labels``).

    Returns:
        The file content.

    Raises:
        _InputError: If the file is missing, a directory, unreadable or not valid UTF-8.
    """
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise _InputError(f"cannot read {what} {path}: {_one_line(exc)}") from exc


def _read_jsonl_source(path: Path) -> _Source:
    """Read a ``frames.jsonl`` file into raw rows.

    Every line that is not a valid frame message is counted as malformed (VL-005), blank
    lines included, as the Spark ingest does.

    Args:
        path: The JSON Lines file.

    Returns:
        The raw rows of all drives in file order and the line counts.

    Raises:
        _InputError: If the file cannot be read.
    """
    lines = _read_text(path, "source").splitlines()
    parsed = [_parse_line(line) for line in lines]
    rows = [row for row in parsed if row is not None]
    raw = pd.DataFrame(rows, columns=RAW_FRAMES_COLUMNS)
    return _Source(raw, len(lines), len(lines) - len(rows))


def _is_parquet_table(path: Path) -> bool:
    """Tell whether a directory looks like a ``raw_frames`` table.

    A table holds Parquet files or at least a ``drive_id=...`` partition directory; a bare
    empty directory is not a table.

    Args:
        path: The directory.

    Returns:
        ``True`` when it holds a ``*.parquet`` file or a ``drive_id=*`` directory.
    """
    return any(path.rglob("*.parquet")) or any(path.glob("drive_id=*"))


def _require_columns(dataset: ds.Dataset, path: Path) -> None:
    """Check that the Parquet files hold every non-optional raw column.

    A table without any file (only ``drive_id=...`` directories) has no columns to check; it
    is an empty table, which the ``source_not_empty`` check reports.

    Args:
        dataset: The discovered dataset.
        path: The table directory, for the error message.

    Raises:
        _InputError: If a required column is absent from the files.
    """
    absent = set(RAW_FRAMES_COLUMNS) - set(METADATA_COLUMNS) - set(dataset.schema.names)
    if dataset.files and absent:
        raise _InputError(f"source {path} has no column {sorted(absent)}")


def _load_parquet(path: Path) -> pd.DataFrame:
    """Load a hive-partitioned Parquet table with ``drive_id`` as a string.

    Args:
        path: The table directory.

    Returns:
        The table content (only the columns present in the files plus ``drive_id``).

    Raises:
        _InputError: If a file is not valid Parquet or the files lack a required column.
    """
    try:
        dataset = ds.dataset(path, format="parquet", partitioning=PARQUET_PARTITIONING)
        _require_columns(dataset, path)
        return pd.DataFrame(dataset.to_table().to_pandas())
    except (OSError, pa.ArrowException) as exc:
        raise _InputError(f"cannot read source {path}: {_one_line(exc)}") from exc


def _read_parquet_source(path: Path) -> _Source:
    """Read a Parquet ``raw_frames`` table directory into raw rows.

    The directory is the table itself (hive partitions ``drive_id=...``); ``drive_id`` is read
    as a string, also for numeric-looking ids. Every row is a valid frame, so ``n_malformed``
    is 0 and ``n_lines`` is the number of rows. The rows are sorted by ``drive_id`` and
    ``RAW_ORDER_COLUMNS`` because the order of the Parquet files is arbitrary: of conflicting
    copies of one ``frame_idx`` the kept one is the first in that order (the lowest ``ts``), not
    the first emitted as for a JSONL source. A table with zero rows is returned as such
    (the ``source_not_empty`` check then fails); a directory that is not a table is an input
    error.

    Args:
        path: The ``raw_frames`` table directory.

    Returns:
        The raw rows of all drives and the row counts.

    Raises:
        _InputError: If the directory cannot be read as a Parquet table.
    """
    if not _is_parquet_table(path):
        raise _InputError(f"cannot read source {path}: no Parquet table (no *.parquet file)")
    raw = _load_parquet(path).reindex(columns=RAW_FRAMES_COLUMNS)
    ordered = raw.sort_values(["drive_id", *RAW_ORDER_COLUMNS], kind="stable", na_position="last")
    return _Source(ordered.reset_index(drop=True), len(raw), 0, "row")


def _read_source(path: Path) -> _Source:
    """Read the source: a Parquet ``raw_frames`` table directory or a ``frames.jsonl`` file.

    Args:
        path: The ``--source`` argument.

    Returns:
        The raw rows and the counts.

    Raises:
        _InputError: If the source cannot be read.
    """
    return _read_parquet_source(path) if path.is_dir() else _read_jsonl_source(path)


def _derive_fleet(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Run ``derive_frames`` on every drive and stack the results.

    Args:
        raw: Raw rows of all drives, in file order.
        cfg: Pipeline thresholds.

    Returns:
        The ``frames`` table of the whole fleet.
    """
    parts = [derive_frames(drive, cfg) for _, drive in raw.groupby("drive_id", sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else derive_frames(raw, cfg)


# --- the report -----------------------------------------------------------------------------


def _with_malformed(
    report: ValidationReport, frames: pd.DataFrame, n_malformed: int, cfg: Config
) -> ValidationReport:
    """Count the malformed lines as VL-005 in ``error_code_counts`` and ``flag_rate``.

    Args:
        report: The report of ``validate_frames``.
        frames: The derived frames.
        n_malformed: Number of malformed lines.
        cfg: Pipeline thresholds (``max_flag_rate``).

    Returns:
        The report with those two checks recomputed over the frames plus the malformed lines.
    """
    rejected = pd.Series([MALFORMED_CODE] * n_malformed, dtype=object)
    codes = pd.concat([frames["error_code"], rejected], ignore_index=True)
    replaced = {
        CHECK_CODE_COUNTS: check_error_code_counts(codes),
        CHECK_FLAG_RATE: check_flag_rate(codes, cfg.max_flag_rate),
    }
    return ValidationReport(tuple(replaced.get(check.name, check) for check in report.checks))


def _label_problem(label: object) -> str | None:
    """Describe what is wrong with one label, if anything.

    Args:
        label: One element of the ``labels`` list.

    Returns:
        A message, or ``None`` when the label is an object with a string ``drive_id``, an
        integer ``frame_idx`` and an ``expected_code`` that is a string or null.
    """
    if not isinstance(label, dict):
        return "a label is not an object"
    index = label.get("frame_idx")
    if not isinstance(label.get("drive_id"), str):
        return "a label has no string drive_id"
    if not isinstance(index, int) or isinstance(index, bool):
        return "a label has no integer frame_idx"
    if "expected_code" not in label or not isinstance(label["expected_code"], (str, type(None))):
        return "a label has no expected_code (a string or null)"
    return None


def _load_labels_json(path: Path) -> object:
    """Read and decode a ``labels.json`` file.

    Args:
        path: The ``labels.json`` file.

    Returns:
        The decoded JSON value.

    Raises:
        _InputError: If the file cannot be read or is not valid JSON.
    """
    try:
        return json.loads(_read_text(path, "labels"))
    except ValueError as exc:
        raise _InputError(f"labels {path} is not valid JSON: {_one_line(exc)}") from exc


def _read_labels(path: Path) -> pd.DataFrame:
    """Read the fault labels of a generated fleet.

    Args:
        path: The ``labels.json`` file (``{"drives": [...], "labels": [...]}``).

    Returns:
        A DataFrame with the ``LABEL_COLUMNS`` columns, one row per label.

    Raises:
        _InputError: If the file cannot be read, is not JSON, has no ``labels`` list, or a
            label is malformed.
    """
    payload = _load_labels_json(path)
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(labels, list):
        raise _InputError(f'labels {path} has no "labels" list')
    problems = [problem for label in labels if (problem := _label_problem(label))]
    if problems:
        raise _InputError(f"labels {path}: {problems[0]}")
    table = pd.DataFrame(labels, columns=LABEL_COLUMNS)
    return table.astype({"drive_id": "object", "frame_idx": "int64"})


def _drive_entry(drive: object) -> tuple[str, tuple[str, str]] | None:
    """Extract ``(drive_id, (profile, vehicle_kind))`` from one ``drives`` element.

    Args:
        drive: One element of the ``drives`` list of ``labels.json``.

    Returns:
        The entry, or ``None`` when the element is not an object with string ``drive_id``,
        ``profile`` and ``vehicle_kind``.
    """
    if not isinstance(drive, dict):
        return None
    drive_id, profile, kind = (drive.get(key) for key in ("drive_id", "profile", "vehicle_kind"))
    if isinstance(drive_id, str) and isinstance(profile, str) and isinstance(kind, str):
        return drive_id, (profile, kind)
    return None


def _read_drives(path: Path) -> dict[str, tuple[str, str]]:
    """Read the profile and vehicle kind of every drive from ``labels.json``.

    Args:
        path: The ``labels.json`` file (``{"drives": [...], "labels": [...]}``).

    Returns:
        ``{drive_id: (profile, vehicle_kind)}`` from the ``drives`` metadata.

    Raises:
        _InputError: If the file cannot be read, is not JSON, has no ``drives`` list, or a
            drive has no string ``drive_id``, ``profile`` and ``vehicle_kind``.
    """
    payload = _load_labels_json(path)
    drives = payload.get("drives") if isinstance(payload, dict) else None
    if not isinstance(drives, list):
        raise _InputError(f'labels {path} has no "drives" list')
    entries = [_drive_entry(drive) for drive in drives]
    if None in entries:
        raise _InputError(f"labels {path}: a drive has no string drive_id, profile, vehicle_kind")
    return dict(entry for entry in entries if entry is not None)


def _safe_id(drive_id: str) -> str:
    r"""Escape a drive id for a report line.

    Args:
        drive_id: A drive id from the data.

    Returns:
        The id with every character that is not printable ASCII (newlines, control characters,
        non-ASCII, lone surrogates) as a backslash escape (``\n``, ``\x1f``, ``\ud800``);
        ordinary ids are unchanged.
    """
    return "".join(
        c if c.isascii() and c.isprintable() else c.encode("unicode_escape").decode()
        for c in drive_id
    )


def check_realism(
    frames: pd.DataFrame, drives: dict[str, tuple[str, str]], cfg: Config
) -> tuple[CheckResult, ...]:
    """Run ``validate_realism`` on every drive of the fleet.

    Args:
        frames: The derived frames of the fleet.
        drives: ``{drive_id: (profile, vehicle_kind)}`` from the labels metadata.
        cfg: Pipeline thresholds.

    Returns:
        The realism checks of all drives in drive order, each named
        ``realism[<drive_id>].<check name>``.

    Raises:
        _InputError: If a drive of the source has no entry in ``drives``.
    """
    checks: list[CheckResult] = []
    for key, drive in frames.groupby("drive_id", sort=False):
        drive_id = str(key)
        if drive_id not in drives:
            raise _InputError(f"drive {drive_id!r} is in the source but not in the labels drives")
        profile, kind = drives[drive_id]
        report = validate_realism(drive.reset_index(drop=True), profile, kind, cfg)
        name = f"realism[{_safe_id(drive_id)}]"
        checks += [replace(c, name=f"{name}.{c.name}") for c in report.checks]
    return tuple(checks)


def check_labeled_faults(frames: pd.DataFrame, labels: pd.DataFrame) -> CheckResult:
    """Check that every labeled fault with an expected code got that code on that frame.

    Labels without an expected code (malformed messages never reach ``frames``) are ignored.

    Args:
        frames: The derived frames of the fleet.
        labels: The fault labels.

    Returns:
        The ``labeled_faults_detected`` result; the detail lists the first misses.
    """
    expected = labels[labels["expected_code"].notna()]
    found = expected.merge(frames[[*KEY, "error_code"]], on=KEY, how="left")
    missed = found[found["error_code"] != found["expected_code"]]
    listed = [
        f"{row.drive_id}#{row.frame_idx} expected {row.expected_code} got {row.error_code}"
        for row in missed.head(MAX_LISTED_MISSES).itertuples()
    ]
    detail = f"{len(found) - len(missed)}/{len(found)} detected" + "".join(f"; {m}" for m in listed)
    return CheckResult(CHECK_LABELED_FAULTS, missed.empty, detail)


def check_false_positives(frames: pd.DataFrame, labels: pd.DataFrame, cfg: Config) -> CheckResult:
    """Check that few frames without a fault label are flagged.

    Args:
        frames: The derived frames of the fleet.
        labels: The fault labels (a frame is labeled when any label names its drive and index).
        cfg: Pipeline thresholds (``max_false_positive_rate``).

    Returns:
        The ``false_positive_rate`` result: flagged share of unlabeled frames is at most
        ``cfg.max_false_positive_rate`` (0 when there are no unlabeled frames).
    """
    keys = labels[KEY].drop_duplicates().assign(labeled=True)
    marked = frames.merge(keys, on=KEY, how="left")
    unlabeled = marked[marked["labeled"].isna()]
    rate = float(unlabeled["error_code"].notna().mean()) if len(unlabeled) else 0.0
    limit = f"limit {cfg.max_false_positive_rate:.2%}"
    detail = f"{rate:.2%} of {len(unlabeled)} unlabeled frames flagged ({limit})"
    return CheckResult(CHECK_FALSE_POSITIVES, rate <= cfg.max_false_positive_rate, detail)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments; ``None`` means ``sys.argv[1:]``.

    Returns:
        The parsed ``source``, ``labels`` and ``realism`` options.
    """
    parser = argparse.ArgumentParser(description="Validate a mock fleet of drives.")
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="frames.jsonl file or raw_frames Parquet table directory to validate",
    )
    parser.add_argument("--labels", type=Path, help="labels.json of the generator")
    parser.add_argument(
        "--realism",
        action="store_true",
        help=(
            "also check each drive against the limits of its profile (needs --labels); "
            "statistical limits that can fail a healthy drive near them, especially a short "
            "one: use at least 1000 frames per drive"
        ),
    )
    return parser.parse_args(argv)


def _source_check(source: _Source) -> CheckResult:
    """Check that the source holds at least one line (JSONL) or row (Parquet).

    Args:
        source: The parsed source.

    Returns:
        The ``source_not_empty`` result; the detail is ``<n> line(s)`` or ``<n> row(s)``.
    """
    detail = f"{source.n_lines} {source.unit}(s)"
    return CheckResult(CHECK_SOURCE_NOT_EMPTY, source.n_lines > 0, detail)


def _duplicates_check(source: _Source, frames: pd.DataFrame) -> CheckResult:
    """Report how many raw rows were collapsed because their ``frame_idx`` repeated.

    ``derive_frames`` keeps one row per distinct ``frame_idx`` of a drive (flagged VL-004), so
    the difference between the raw rows and the frames is the number of rows dropped.

    Args:
        source: The parsed source.
        frames: The derived frames.

    Returns:
        The ``duplicates_collapsed`` result (always passing); the detail is ``<n> row(s)``.
    """
    return CheckResult(CHECK_DUPLICATES, True, f"{len(source.raw) - len(frames)} row(s)")


def _frames_lost_check(frames: pd.DataFrame) -> CheckResult:
    """Report how many frames are missing inside each drive's ``frame_idx`` sequence.

    VL-001 flags one row per gap; this counts the frames each gap lost (``step - 1``). Frames
    lost after the last received frame of a drive cannot be seen.

    Args:
        frames: The derived frames (sorted by ``frame_idx`` within each drive).

    Returns:
        The ``frames_lost`` result (always passing); the detail is
        ``<n> frame(s) in <m> gap(s)``.
    """
    steps = frames.groupby("drive_id", sort=False)["frame_idx"].diff()
    gaps = steps[steps > 1]
    detail = f"{int((gaps - 1).sum())} frame(s) in {len(gaps)} gap(s)"
    return CheckResult(CHECK_FRAMES_LOST, True, detail)


def _build_report(args: argparse.Namespace, cfg: Config) -> ValidationReport:
    """Read the inputs and run every check.

    Args:
        args: The parsed command line.
        cfg: Pipeline thresholds.

    Returns:
        The full report.

    Raises:
        _InputError: If the source or the labels file cannot be used, ``--realism`` is given
            without ``--labels``, or a drive of the source has no entry in the labels
            ``drives``. (An invalid ``VL_*`` value is raised by ``main`` before this call.)
    """
    if args.realism and args.labels is None:
        raise _InputError(REALISM_NEEDS_LABELS)
    source = _read_source(args.source)
    labels = _read_labels(args.labels) if args.labels is not None else None
    drives = _read_drives(args.labels) if args.realism else None
    frames = _derive_fleet(source.raw, cfg)
    report = _with_malformed(validate_frames(frames, cfg), frames, source.n_malformed, cfg)
    checks = (
        _source_check(source),
        _duplicates_check(source, frames),
        _frames_lost_check(frames),
        *report.checks,
    )
    if labels is not None:
        checks += (check_labeled_faults(frames, labels), check_false_positives(frames, labels, cfg))
    if drives is not None:
        checks += check_realism(frames, drives, cfg)
    return ValidationReport(checks)


def _load_cfg() -> Config:
    """Load the pipeline thresholds from the ``VL_*`` environment variables.

    Returns:
        The validated config.

    Raises:
        _InputError: If a ``VL_*`` value is invalid; the message is the one of ``load_config``.
    """
    try:
        return load_config()
    except ValueError as exc:
        raise _InputError(_one_line(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Run the validator command line.

    ``--source`` is a ``frames.jsonl`` file or the directory of a Parquet ``raw_frames`` table
    (hive partitions ``drive_id=...``, ``drive_id`` read as a string); every drive goes through
    ``derive_frames`` and ``validate_frames``, and the combined report is printed to stdout. A
    ``source_not_empty`` check comes first and fails on a file without lines or a Parquet table
    without rows (detail ``<n> row(s)``; a bare empty directory is not a table and exits ``2``).
    Two informational lines follow and always pass: ``duplicates_collapsed`` (raw rows dropped
    because their ``frame_idx`` repeated) and ``frames_lost`` (frames missing inside the
    ``frame_idx`` sequences, with the number of gaps). Lines that are not valid frame messages
    count as ``VL-005`` in ``error_code_counts`` and in ``flag_rate`` (so the 5% limit governs);
    a message with an
    empty ``points`` list is a normal frame flagged ``VL-006``. With ``--labels`` (the
    ``labels.json`` of the generator) two more checks run: ``labeled_faults_detected`` (every
    labeled fault with an expected code got that code on that frame) and
    ``false_positive_rate`` (flagged share of unlabeled frames is within
    ``cfg.max_false_positive_rate``). With ``--realism`` (which needs ``--labels``) every drive
    of the source also goes through ``validate_realism`` with the ``profile`` and
    ``vehicle_kind`` of its entry in the ``drives`` metadata of ``labels.json``; the checks are
    appended as ``realism[<drive_id>].<check name>`` (the id escaped if it holds control or
    non-ASCII characters). These are the ``PROFILE_LIMITS`` maximum and p99 checks: a healthy
    drive near a limit can fail (1 of 40 default fleets: a parking drive, ``accel_p99`` 2.003
    against 2.0) and short drives (about 300 frames) give noisy p99 estimates; use at least 1000
    frames per drive.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when every check passes, ``1`` when a check fails, ``2`` when the source or the
        labels file cannot be used (missing, a directory, undecodable, malformed), ``--realism``
        lacks ``--labels``, a source drive has no ``drives`` metadata, or a ``VL_*`` environment
        value is invalid (the reason is one line on stderr). Argparse usage errors also exit
        with ``2``.
    """
    args = _parse_args(argv)
    try:
        report = _build_report(args, _load_cfg())
    except _InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    print(report.to_text())
    return 0 if report.passed else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
