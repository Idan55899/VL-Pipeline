"""Load the committed golden fixtures as typed objects and DataFrames.

Role in the pipeline: test support only. The unit tests for ``derive_frames``,
``detect_events`` and ``summarize_drive`` read their inputs and expected values here.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

GOLDEN_DIR = Path(__file__).parent / "golden"
RAW_DTYPES: dict[str, str] = {
    "drive_id": "object",
    "frame_idx": "int64",
    "ts": "float64",
    "n_points": "int64",
    "x": "float64",
    "y": "float64",
    "z": "float64",
}


@dataclass(frozen=True)
class GoldenCase:
    """One golden mock drive with its analytically derived expectations.

    Attributes:
        case: Case name, equal to the JSON file stem.
        description: Human-readable summary of the scenario.
        raw: Raw frame rows (``drive_id, frame_idx, ts, n_points, x, y, z``).
        expected: Expected results; keys are optional per case.
    """

    case: str
    description: str
    raw: list[dict[str, Any]]
    expected: dict[str, Any]


def list_golden_cases() -> list[str]:
    """List the available golden case names.

    Returns:
        Sorted case names (JSON file stems in the golden directory).
    """
    return sorted(path.stem for path in GOLDEN_DIR.glob("*.json"))


def load_golden(case: str) -> GoldenCase:
    """Load one golden case from disk.

    Args:
        case: Case name such as ``"g01_constant_speed"``.

    Returns:
        The parsed `GoldenCase`.

    Raises:
        FileNotFoundError: If no fixture file exists for ``case``.
    """
    data = json.loads((GOLDEN_DIR / f"{case}.json").read_text())
    return GoldenCase(data["case"], data["description"], data["raw"], data["expected"])


def raw_dataframe(case: GoldenCase) -> pd.DataFrame:
    """Build the raw-frame DataFrame for a case.

    Args:
        case: The golden case.

    Returns:
        A frame with columns exactly ``RAW_DTYPES`` (empty cases keep the dtypes).
    """
    frame = pd.DataFrame(case.raw, columns=list(RAW_DTYPES))
    return frame.astype(RAW_DTYPES)
