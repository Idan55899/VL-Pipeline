"""Tests for the dashboard helpers and app (``vl.dashboard.app``) on real small warehouses.

The golden warehouse is the session-wide transform output (``golden_output``); the small one is
hand-written Parquet with vehicle kinds and rejects; the empty one is an empty directory.
"""

import os
import subprocess
import sys
from pathlib import Path

import dash
import pandas as pd
import plotly.graph_objects as go
import pytest

from fixtures.loader import list_golden_cases, load_golden
from transform_helpers import GoldenOutput
from vl.dashboard.app import create_app, drive_options, drive_stats, drive_view, quality_view
from vl.schemas import DRIVES_COLUMNS, DRIVES_DTYPES
from vl.warehouse import Warehouse

SRC = Path(__file__).resolve().parent.parent / "src"
PRESENT = [n for n in list_golden_cases() if load_golden(n).expected["n_frames_out"] > 0]


@pytest.fixture(scope="module")
def golden_wh(golden_output: GoldenOutput) -> Warehouse:
    """A warehouse over the transform output of all golden cases."""
    return Warehouse(str(golden_output.warehouse))


@pytest.fixture
def empty_wh(tmp_path: Path) -> Warehouse:
    """A warehouse over an empty directory."""
    return Warehouse(str(tmp_path))


@pytest.fixture
def small_wh(tmp_path: Path) -> Warehouse:
    """A warehouse with two drives of different vehicle kinds and two rejects."""
    rows = [
        ("d1", "veh-1", "car", "2026-01-01T00:00:00+00:00", 0.0, 10.0, 160, 100.0, 0, 0),
        ("d2", "veh-2", "truck", "2026-01-02T00:00:00+00:00", 0.0, 12.0, 192, 90.0, 0, 0),
    ]
    drives = pd.DataFrame(rows, columns=DRIVES_COLUMNS).astype(DRIVES_DTYPES)
    rejects = pd.DataFrame(
        {
            "payload": ["{bad", "[]"],
            "error_code": "VL-005",
            "ingested_at": pd.Timestamp("2026-01-01"),
        }
    )
    for name, table in (("drives", drives), ("rejects", rejects)):
        (tmp_path / name).mkdir()
        table.to_parquet(tmp_path / name / "part.parquet")
    return Warehouse(str(tmp_path))


def golden_error_counts() -> dict[str, int]:
    """Count the golden error codes over all drives, from the fixtures' ``expected`` keys."""
    counts: dict[str, int] = {}
    for name in PRESENT:
        for code in load_golden(name).expected["error_codes"].values():
            counts[code] = counts.get(code, 0) + 1
    return counts


def shows_no_data(fig: go.Figure) -> bool:
    """Tell whether the figure carries a "No data" annotation."""
    return any("No data" in (a.text or "") for a in fig.layout.annotations)


def test_helpers_on_populated_warehouses(golden_wh: Warehouse, small_wh: Warehouse) -> None:
    options = drive_options(small_wh)
    assert [o["value"] for o in options] == ["d1", "d2"]
    assert "d1" in options[0]["label"] and "car" in options[0]["label"]
    assert "d2" in options[1]["label"] and "truck" in options[1]["label"]
    assert [o["value"] for o in drive_options(golden_wh)] == PRESENT
    trajectory, speed, rows = drive_view(golden_wh, "g02_hard_brake")
    line = [t for t in trajectory.data if t.mode != "markers"]
    assert len(line) == 1 and len(line[0].x) == 160  # g02 has 160 frames, all with a position
    assert len(speed.layout.shapes) == 1
    assert [r["event_type"] for r in rows] == ["HARD_BRAKE"]
    figure, rejects = quality_view(golden_wh)
    assert dict(zip(figure.data[0].x, figure.data[0].y)) == golden_error_counts()
    assert rejects == 0 and quality_view(small_wh)[1] == 2
    assert drive_stats(small_wh, "d1") == [
        ("Vehicle", "car"),
        ("Frames", "160"),
        ("Duration", "10.0 s"),
        ("Distance", "100 m"),
        ("Events", "0"),
        ("Flagged", "0"),
    ]


def test_helpers_on_an_empty_warehouse(empty_wh: Warehouse) -> None:
    assert drive_options(empty_wh) == []
    for chosen in (None, "no-such-drive"):
        trajectory, speed, rows = drive_view(empty_wh, chosen)
        assert shows_no_data(trajectory) and shows_no_data(speed), chosen
        assert rows == [], chosen
        assert drive_stats(empty_wh, chosen) == [], chosen
    figure, rejects = quality_view(empty_wh)
    assert shows_no_data(figure) and rejects == 0


def test_create_app_has_both_tabs(empty_wh: Warehouse, golden_wh: Warehouse) -> None:
    for name, wh in (("empty", empty_wh), ("golden", golden_wh)):
        app = create_app(wh)
        assert isinstance(app, dash.Dash), name
        layout = str(app.layout)
        assert "Scenario Explorer" in layout and "Data Quality" in layout, name


def test_figures_import_does_not_need_dash() -> None:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    code = "import sys, vl.dashboard.figures; sys.exit(3 if 'dash' in sys.modules else 0)"
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert done.returncode == 0, f"dash was imported or the import failed:\n{done.stderr}"
