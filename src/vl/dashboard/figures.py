"""Pure Plotly figure builders of the dashboard.

Role in the pipeline: the drawing step of the presentation layer. Every function takes pandas
DataFrames (rows of the ``frames``, ``events`` tables or of an example query) and returns a
``plotly.graph_objects.Figure``; nothing here imports Dash or touches the warehouse.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vl.errors import EventType

INK = "#111a21"
MUTE = "#5d6b75"
GRID = "#e4eaee"
RULE = "#ccd5db"
TRACE = "#0f7b78"  # speed, and the dashboard's accent
ACCEL = "#4a5a66"
PATH = "#a3b2bc"  # the driven path is quiet, so the event markers carry the eye
SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"
# One colour and one symbol per event type, so a type looks the same in every drive and the
# shape alone still tells them apart without colour. Each symbol mimics its kinematics.
EVENT_STYLE = {
    EventType.HARD_BRAKE.value: ("#a4162c", "triangle-down"),
    EventType.HARD_ACCEL.value: ("#9a5a00", "triangle-up"),
    EventType.SHARP_TURN.value: ("#6d3f9c", "diamond"),
    EventType.STOP.value: (TRACE, "square"),
}
OTHER_STYLE = (ACCEL, "circle")
SPAN_COLOR = "rgba(17, 26, 33, 0.06)"
# Static styling of an event's shaded span and label; only the times and the text vary.
SPAN_STYLE = {"type": "rect", "xref": "x", "yref": "paper", "y0": 0, "y1": 1, "line_width": 0}
LABEL_STYLE = {"xref": "x", "yref": "paper", "xanchor": "left", "yanchor": "top"}
# Two events close in time would print their labels into each other, so consecutive labels
# alternate between these heights.
LABEL_ROWS = (1.0, 0.93)
MAX_POINTS = 20000  # a longer drive is drawn evenly sampled, with a note in the figure
SMALL_COUNT = 10  # below this largest count, the count axis ticks every 1 (whole frames only)
# Readings are monospace and axis furniture is quiet; the card around the figure carries the
# heading, so the figure keeps no title of its own and no top margin for one.
AXIS_STYLE = {
    "gridcolor": GRID,
    "zerolinecolor": RULE,
    "linecolor": RULE,
    "tickfont": {"family": MONO, "size": 11, "color": MUTE},
    "title_font": {"size": 11, "color": MUTE},
}
SPEED_LINE = {"color": TRACE, "width": 2}
ACCEL_LINE = {"color": ACCEL, "width": 1.4, "dash": "dot"}  # quieter: it has its own axis
SPEED_AXIS = {"title": {"text": "speed (m/s)"}}
# The right-hand axis is created by ``update_layout``, which AXIS_STYLE never reaches, so it
# spells the shared styling out.
ACCEL_AXIS = {
    "title": {"text": "accel (m/s^2)", "font": {"size": 11, "color": MUTE}},
    "tickfont": {"family": MONO, "size": 11, "color": MUTE},
    # An overlaying axis takes its tick count from the left one, so the values land on
    # uneven numbers; one decimal each at least keeps the column even.
    "tickformat": ".1f",
    "overlaying": "y",
    "side": "right",
    "showgrid": False,
    "linecolor": RULE,
}


def _base() -> go.Figure:
    """Build an empty figure with the shared look: transparent paper, hairline axes.

    Returns:
        The figure without traces.
    """
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0, 0, 0, 0)",  # the card behind the figure provides the surface
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"family": SANS, "size": 12, "color": INK},
        margin={"l": 58, "r": 58, "t": 8, "b": 44},
        legend={"orientation": "h", "y": -0.22, "font": {"size": 11}},
        hoverlabel={"font": {"family": MONO, "size": 12}},
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


def empty_fig(message: str) -> go.Figure:
    """Build a figure that shows only a centered message.

    Args:
        message: The text of the centered annotation, for example ``"No data"``.

    Returns:
        A figure with no traces, hidden axes and one annotation holding the message.
    """
    fig = _base()
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"family": MONO, "size": 15, "color": MUTE},
    )
    return fig


def _event_positions(positioned: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach to every event the position of its start frame (events without one are dropped).

    Args:
        positioned: Frames with a position.
        events: ``events`` rows.

    Returns:
        The events with the columns ``x`` and ``y`` added.
    """
    at = positioned[["frame_idx", "x", "y"]].rename(columns={"frame_idx": "start_frame"})
    return events.merge(at, on="start_frame", how="inner")


def _thin(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep at most ``MAX_POINTS`` evenly spaced rows of a long drive.

    Args:
        frame: Frames sorted in time.

    Returns:
        The frame itself when short enough, else an even subsample.
    """
    if len(frame) <= MAX_POINTS:
        return frame
    return frame.iloc[np.linspace(0, len(frame) - 1, MAX_POINTS).astype(int)]


def _note(fig: go.Figure, shown: int, total: int) -> None:
    """Add a small note to the figure when its traces were thinned out.

    Args:
        fig: The figure to annotate.
        shown: Number of plotted points.
        total: Number of frames in the drive.
    """
    if shown < total:
        fig.add_annotation(
            text=f"Showing {shown} of {total} frames (evenly sampled)",
            x=1,
            y=-0.3,
            xref="paper",
            yref="paper",
            xanchor="right",
            showarrow=False,
            font={"size": 11, "color": MUTE},
        )


def _event_style(kind: object) -> tuple[str, str]:
    """Look up the colour and marker symbol of one event type.

    Args:
        kind: An ``event_type`` value.

    Returns:
        The type's ``(colour, symbol)``, or the fallback pair for a type added later.
    """
    return EVENT_STYLE.get(str(kind), OTHER_STYLE)


def _marker_traces(marked: pd.DataFrame) -> list[go.Scatter]:
    """Build one marker trace per event type, each in that type's own colour and symbol.

    Args:
        marked: Events with the columns ``x`` and ``y``.

    Returns:
        The traces (``mode="markers"``) ordered by event type.
    """
    traces: list[go.Scatter] = []
    for kind, group in marked.groupby("event_type", sort=True):
        color, symbol = _event_style(kind)
        rim = {"width": 1.5, "color": "white"}  # keeps a marker legible on top of the path
        marker = {"size": 11, "color": color, "symbol": symbol, "line": rim}
        traces.append(
            go.Scatter(x=group["x"], y=group["y"], mode="markers", name=str(kind), marker=marker)
        )
    return traces


def trajectory_fig(frames: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    """Draw the x-y path of one drive with its events marked.

    One line trace holds a point per frame that has a position; one marker trace per distinct
    event type (event markers use ``mode="markers"``) marks the position of each event's start
    frame. The axes have equal scale. A drive of more than ``MAX_POINTS`` frames is drawn
    evenly sampled, with a note in the figure.

    Args:
        frames: ``frames`` rows of one drive.
        events: ``events`` rows of the same drive (may be empty).

    Returns:
        The figure, or ``empty_fig("No data")`` when ``frames`` is empty.
    """
    if frames.empty:
        return empty_fig("No data")
    positioned = frames.dropna(subset=["x", "y"]).sort_values("frame_idx")
    thin = _thin(positioned)
    fig = _base()
    path = go.Scatter(
        x=thin["x"], y=thin["y"], mode="lines", name="path", line={"color": PATH, "width": 2}
    )
    fig.add_trace(path)
    fig.add_traces(_marker_traces(_event_positions(positioned, events)))
    fig.update_xaxes(title_text="x (m)")
    fig.update_yaxes(title_text="y (m)", scaleanchor="x", scaleratio=1)
    _note(fig, len(thin), len(positioned))
    return fig


def _event_spans(events: pd.DataFrame) -> tuple[list[go.layout.Shape], list[go.layout.Annotation]]:
    """Build the shaded span and the label of every event.

    Args:
        events: ``events`` rows.

    Returns:
        The shapes and annotations, one of each per event, in row order.
    """
    rows = list(events.itertuples())
    shapes = [
        go.layout.Shape(
            x0=r.start_ts, x1=r.end_ts, fillcolor=SPAN_COLOR, layer="below", **SPAN_STYLE
        )
        for r in rows
    ]
    labels = [
        go.layout.Annotation(
            text=str(r.event_type),
            x=r.start_ts,
            y=LABEL_ROWS[i % len(LABEL_ROWS)],
            showarrow=False,
            # same colour as the type's marker in the trajectory, which links the two figures
            font={"family": MONO, "size": 10, "color": _event_style(r.event_type)[0]},
            **LABEL_STYLE,
        )
        for i, r in enumerate(rows)
    ]
    return shapes, labels


def speed_fig(frames: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    """Draw speed and acceleration of one drive versus time, with its events shaded.

    Speed uses the left axis and acceleration the right one. The event spans are built as
    lists and set once (one ``add_vrect`` per event is quadratic in the number of events).

    Args:
        frames: ``frames`` rows of one drive.
        events: ``events`` rows of the same drive; each becomes a shaded vertical span from
            ``start_ts`` to ``end_ts`` labelled with its type.

    Returns:
        The figure with a ``speed`` and an ``accel`` trace, or ``empty_fig("No data")`` when
        ``frames`` is empty.
    """
    if frames.empty:
        return empty_fig("No data")
    ordered = frames.sort_values("ts")
    thin = _thin(ordered)
    fig = _base()
    fig.add_trace(
        go.Scatter(x=thin["ts"], y=thin["speed"], mode="lines", name="speed (m/s)", line=SPEED_LINE)
    )
    accel = go.Scatter(x=thin["ts"], y=thin["accel"], mode="lines", name="accel (m/s^2)")
    fig.add_trace(accel.update(yaxis="y2", line=ACCEL_LINE))
    shapes, labels = _event_spans(events)
    fig.update_layout(shapes=shapes, annotations=labels, yaxis=SPEED_AXIS, yaxis2=ACCEL_AXIS)
    fig.update_xaxes(title_text="time (s)")
    _note(fig, len(thin), len(ordered))
    return fig


def error_counts_fig(counts: pd.DataFrame) -> go.Figure:
    """Draw a bar chart of the number of frames per error code.

    Args:
        counts: Rows with the columns ``error_code`` and ``n_frames``.

    Returns:
        The bar chart in the row order of ``counts``, or ``empty_fig("No data")`` when
        ``counts`` is empty.
    """
    if counts.empty:
        return empty_fig("No data")
    fig = _base()
    bar = go.Bar(x=counts["error_code"], y=counts["n_frames"], name="frames")
    fig.add_trace(bar.update(marker={"color": TRACE}, width=0.38))
    fig.update_xaxes(title_text="error code")
    fig.update_yaxes(title_text="frames (count)", tickformat=",d", rangemode="tozero")
    if counts["n_frames"].max() < SMALL_COUNT:  # Plotly would tick at 0.5 and repeat labels
        fig.update_yaxes(dtick=1)
    fig.update_layout(showlegend=False)
    return fig
