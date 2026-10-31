"""Bookkeeping of the incremental transform: which drives changed since the last run.

Role in the pipeline: lets `vl.etl.transform` recompute only what is new. The ingest appends
Parquet files to ``raw_frames/drive_id=<id>/`` and never edits one, so a drive's partition is
unchanged exactly when its file names and sizes are. After every run the transform saves those
signatures, with the thresholds it used, in ``<warehouse>/_state/transform.json``. The next
incremental run compares: new or changed partitions are recomputed, vanished ones are removed,
the rest is left alone. A full run is needed when there is no state yet, when the thresholds or
the pipeline code changed (every derived value may change) or when an output table is missing.

Drives are identified by their partition directory name (``drive_id=<escaped id>``); Spark
writes ``raw_frames`` and ``frames`` with the same escaping, so the name is the same in both.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote

from vl.config import Config

STATE_FILE = Path("_state") / "transform.json"
OUTPUTS = ("frames", "events", "drives")
_PREFIX = "drive_id="
_PACKAGE = Path(__file__).resolve().parents[1]
# The modules whose code decides what frames, events and drives hold. Editing one changes the
# outputs of every drive, even when no raw file and no threshold changed.
CODE_FILES = tuple(
    _PACKAGE / name
    for name in (
        "etl/features.py",
        "etl/transform.py",
        "validation.py",
        "schemas.py",
        "errors.py",
        "config.py",
    )
)


class Plan(NamedTuple):
    """What an incremental run has to do.

    Attributes:
        changed: Partition directory names of drives that are new or whose raw files changed.
        removed: Partition directory names of drives that are no longer in ``raw_frames``.
    """

    changed: list[str]
    removed: list[str]


def raw_signature(warehouse: Path) -> dict[str, list[list[object]]]:
    """Describe every ``raw_frames`` partition by its files.

    Args:
        warehouse: Warehouse directory.

    Returns:
        ``{partition directory name: [[file name, size], ...]}``, files sorted by name.
    """
    root = warehouse / "raw_frames"
    parts = sorted(p for p in root.glob(f"{_PREFIX}*") if p.is_dir()) if root.is_dir() else []
    return {
        p.name: [[f.name, f.stat().st_size] for f in sorted(p.glob("*.parquet"))] for p in parts
    }


def _config_key(cfg: Config) -> str:
    """Serialize the thresholds so two runs can be compared.

    Args:
        cfg: Pipeline thresholds.

    Returns:
        A stable JSON string.
    """
    return json.dumps(asdict(cfg), sort_keys=True)


def _code_key() -> str:
    """Fingerprint the pipeline code that produces the output tables.

    Returns:
        The SHA-256 hex digest of the files in `CODE_FILES`, in order.
    """
    digest = hashlib.sha256()
    for path in CODE_FILES:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def plan(warehouse: Path, cfg: Config) -> Plan | None:
    """Work out what changed since the last run.

    Args:
        warehouse: Warehouse directory.
        cfg: The thresholds of this run.

    Returns:
        The changed and removed drives, or ``None`` when a full run is needed (no readable
        state, other thresholds, changed pipeline code, or a missing output table).
    """
    path = warehouse / STATE_FILE
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    same_logic = state.get("config") == _config_key(cfg) and state.get("code") == _code_key()
    if not same_logic or not all((warehouse / t).is_dir() for t in OUTPUTS):
        return None
    old, new = state.get("drives", {}), raw_signature(warehouse)
    changed = [name for name, files in new.items() if old.get(name) != files]
    return Plan(changed, [name for name in old if name not in new])


def save_state(warehouse: Path, cfg: Config) -> None:
    """Record the current raw signatures, thresholds and code after a successful run.

    Args:
        warehouse: Warehouse directory.
        cfg: The thresholds the run used.
    """
    path = warehouse / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"config": _config_key(cfg), "code": _code_key(), "drives": raw_signature(warehouse)}
    path.write_text(json.dumps(state, indent=1))


def drive_id_of(partition: str) -> str:
    """Recover the drive id from a partition directory name.

    Args:
        partition: ``drive_id=<escaped id>`` (Spark escapes special characters as ``%XX``).

    Returns:
        The drive id.
    """
    return unquote(partition.removeprefix(_PREFIX))
