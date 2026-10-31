"""Tests for the pure Plotly figure builders (``vl.dashboard.figures``).

Expectations are hand-derived from small hand-made drives: ``frames_table`` gives frames at
``ts = i / 16`` s on the line ``x = 10 * ts``, ``y = 0``.
"""

import pandas as pd
import plotly.graph_objects as go
import pytest

from pipeline_helpers import DT, frames_table
from vl.dashboard.figures import error_counts_fig, speed_fig, trajectory_fig
from vl.schemas import EVENTS_COLUMNS, EVENTS_DTYPES

# (event_type, start_frame, end_frame) of the two hand-made events of a 5-frame drive.
EVENT_SPECS = [("HARD_BRAKE", 1, 2), ("STOP", 3, 4)]


def make_events(specs: list[tuple[str, int, int]]) -> pd.DataFrame:
    """Build an ``events`` table whose timestamps are ``frame_idx * DT``."""
    rows = [("d", kind, s, e, s * DT, e * DT, "{}") for kind, s, e in specs]
    return pd.DataFrame(rows, columns=EVENTS_COLUMNS).astype(EVENTS_DTYPES)


def shows_no_data(fig: go.Figure) -> bool:
    """Tell whether the figure carries a "No data" annotation."""
    return any("No data" in (a.text or "") for a in fig.layout.annotations)


def test_trajectory_fig_line_and_event_markers() -> None:
    frames, events = frames_table(5), make_events(EVENT_SPECS)
    fig = trajectory_fig(frames, events)
    markers = [t for t in fig.data if t.mode == "markers"]
    lines = [t for t in fig.data if t.mode != "markers"]
    assert len(lines) == 1 and len(lines[0].x) == 5 and len(lines[0].y) == 5
    assert list(lines[0].x) == pytest.approx(list(frames["x"]))
    # One marker trace per distinct event type, at the position of the event's start frame
    # (frame 1: x = 10 / 16 = 0.625 m; frame 3: x = 30 / 16 = 1.875 m; y = 0).
    assert len(markers) == len(EVENT_SPECS)
    points = sorted((float(t.x[0]), float(t.y[0])) for t in markers)
    assert points == pytest.approx([(0.625, 0.0), (1.875, 0.0)])
    assert fig.layout.yaxis.scaleanchor == "x"  # equal axis scale
    assert shows_no_data(trajectory_fig(frames.iloc[0:0], make_events([])))


def test_speed_fig_traces_and_event_spans() -> None:
    speed, accel = [1.0, 2.0, 3.0, 4.0, 5.0], [0.5, -1.0, -2.0, -3.0, 0.0]
    frames = frames_table(5, speed=speed, accel=accel)
    fig = speed_fig(frames, make_events(EVENT_SPECS))
    assert len(fig.data) == 2
    assert fig.layout.yaxis.title.text == "speed (m/s)"
    assert fig.layout.yaxis2.title.text == "accel (m/s^2)"
    assert sorted(tuple(t.y) for t in fig.data) == sorted([tuple(speed), tuple(accel)])
    for trace in fig.data:
        assert list(trace.x) == pytest.approx(list(frames["ts"]))
    spans = sorted((s.x0, s.x1) for s in fig.layout.shapes)
    assert spans == pytest.approx([(1 * DT, 2 * DT), (3 * DT, 4 * DT)])
    labels = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "HARD_BRAKE" in labels and "STOP" in labels
    assert len(speed_fig(frames, make_events([])).layout.shapes) == 0  # drive without events
    assert shows_no_data(speed_fig(frames.iloc[0:0], make_events([])))


def test_error_counts_fig_bars_match_counts() -> None:
    counts = pd.DataFrame({"error_code": ["VL-001", "VL-002", "VL-004"], "n_frames": [2, 1, 3]})
    fig = error_counts_fig(counts)
    assert len(fig.data) == 1 and fig.data[0].type == "bar"
    assert list(fig.data[0].x) == ["VL-001", "VL-002", "VL-004"]
    assert list(fig.data[0].y) == [2, 1, 3]
    assert fig.layout.yaxis.dtick == 1  # small counts: whole-number ticks, no repeated labels
    assert shows_no_data(error_counts_fig(counts.iloc[0:0]))
