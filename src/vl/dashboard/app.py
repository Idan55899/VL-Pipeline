"""Dash application: scenario explorer and data-quality tabs.

Role in the pipeline: the last stage. It reads the warehouse only through
``Warehouse.query`` and draws with ``vl.dashboard.figures``. Dash is imported lazily so the
helpers and the figures work without it.
"""

import argparse
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from vl.dashboard.figures import empty_fig, error_counts_fig, speed_fig, trajectory_fig
from vl.errors import ErrorCode, EventType
from vl.warehouse import EXAMPLE_QUERIES, Warehouse

if TYPE_CHECKING:
    import dash
    from dash.development.base_component import Component

EVENT_COLUMNS = ["event_type", "start_frame", "end_frame", "start_ts", "end_ts", "params"]
SUBTITLE = "Scenario mining over the Vehicle Location signal, with a quality code on every frame."
# Tell the browser the page is light (native scrollbars and controls follow) and colour the
# mobile browser chrome to match the dark header band.
META_TAGS = [
    {"name": "color-scheme", "content": "light"},
    {"name": "theme-color", "content": "#111a21"},
    {"name": "viewport", "content": "width=device-width, initial-scale=1"},
]
NO_DATA_NOTICE = "No drives in this warehouse. Run make demo first."
NO_DRIVE_NOTICE = "Choose a drive to see its readings."
MAX_CROSS_ROWS = 500
# Only the columns the two drive figures draw: Parquet is columnar, so the other columns
# (vx, vy, yaw_rate, z, metadata) are never read from disk.
DRIVE_FRAMES_SQL = (
    "SELECT frame_idx, ts, x, y, speed, accel, error_code FROM frames "
    "WHERE drive_id = ? ORDER BY frame_idx"
)
ERRORS_PER_DRIVE_SQL = (
    "SELECT drive_id, error_code, COUNT(*) AS n_frames FROM frames "
    "WHERE error_code IS NOT NULL GROUP BY drive_id, error_code ORDER BY drive_id, error_code"
)
DRIVE_STATS_SQL = (
    "SELECT vehicle_kind, n_frames, duration_s, distance_m, n_events, n_errors "
    "FROM drives WHERE drive_id = ?"
)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a query result to table rows.

    Args:
        frame: Any query result.

    Returns:
        One dict per row, keyed by column name.
    """
    columns = [str(name) for name in frame.columns]
    return [dict(zip(columns, row, strict=True)) for row in frame.itertuples(index=False)]


def drive_options(wh: Warehouse) -> list[dict[str, str]]:
    """List the drives for the drive dropdown.

    Args:
        wh: The warehouse.

    Returns:
        One ``{"label", "value"}`` dict per drive in ``drives``, ordered by drive id; the
        value is the drive id and the label shows the drive id with its vehicle kind. Empty
        for an empty warehouse.
    """
    drives = wh.query("SELECT drive_id, vehicle_kind FROM drives ORDER BY drive_id")
    ids = [str(d) for d in drives["drive_id"]]
    kinds = [str(k) if isinstance(k, str) else "unknown kind" for k in drives["vehicle_kind"]]
    return [{"label": f"{d} ({k})", "value": d} for d, k in zip(ids, kinds, strict=True)]


def drive_view(
    wh: Warehouse, drive_id: str | None
) -> tuple[go.Figure, go.Figure, list[dict[str, object]]]:
    """Build everything the scenario explorer shows for one drive.

    Args:
        wh: The warehouse.
        drive_id: The chosen drive, or None when nothing is chosen.

    Returns:
        The trajectory figure, the speed figure and the events table rows; empty figures
        ("No data") and no rows when the drive is None or unknown.
    """
    if drive_id is None:
        return empty_fig("No data"), empty_fig("No data"), []
    frames = wh.query(DRIVE_FRAMES_SQL, [drive_id])
    events = wh.query("SELECT * FROM events WHERE drive_id = ? ORDER BY start_frame", [drive_id])
    return trajectory_fig(frames, events), speed_fig(frames, events), _records(events)


def drive_stats(wh: Warehouse, drive_id: str | None) -> list[tuple[str, str]]:
    """Read the chosen drive's summary row as labelled readings.

    The drive dropdown names a drive but says nothing about it; these are the numbers the
    ``drives`` table already holds, so picking a drive answers "how long, how far, how
    clean" without a query.

    Args:
        wh: The warehouse.
        drive_id: The chosen drive, or None when nothing is chosen.

    Returns:
        ``(label, value)`` pairs in display order, the values already formatted; empty when
        the drive is None or absent from ``drives``.
    """
    if drive_id is None:
        return []
    rows = wh.query(DRIVE_STATS_SQL, [drive_id])
    if rows.empty:
        return []
    row = rows.iloc[0]
    return [
        ("Vehicle", str(row["vehicle_kind"])),
        ("Frames", f"{int(row['n_frames']):,}"),
        ("Duration", f"{float(row['duration_s']):.1f} s"),
        ("Distance", f"{float(row['distance_m']):,.0f} m"),
        ("Events", f"{int(row['n_events'])}"),
        ("Flagged", f"{int(row['n_errors'])}"),  # frames carrying an error code
    ]


def quality_view(wh: Warehouse) -> tuple[go.Figure, int]:
    """Build the data-quality tab content.

    Args:
        wh: The warehouse.

    Returns:
        The error-count figure (from ``EXAMPLE_QUERIES['error_counts']``) and the number of
        rows in ``rejects``; an empty figure and 0 for an empty warehouse.
    """
    counts = wh.query(EXAMPLE_QUERIES["error_counts"])
    rejects = wh.query("SELECT COUNT(*) AS n FROM rejects")
    return error_counts_fig(counts), int(rejects["n"].iloc[0])


def _filtered_events(wh: Warehouse, kinds: Sequence[str]) -> list[dict[str, object]]:
    """Fetch the events of all drives, restricted to some event types.

    Args:
        wh: The warehouse.
        kinds: The selected event types; none selected gives no rows.

    Returns:
        Up to ``MAX_CROSS_ROWS`` table rows ordered by drive and start time.
    """
    if not kinds:
        return []
    marks = ", ".join("?" for _ in kinds)
    sql = (
        f"SELECT drive_id, {', '.join(EVENT_COLUMNS)} FROM events WHERE event_type IN ({marks}) "
        f"ORDER BY drive_id, start_ts LIMIT {MAX_CROSS_ROWS}"
    )
    return _records(wh.query(sql, list(kinds)))


def _format(value: object) -> str:
    """Format a table cell: floats to 4 decimals, JSON objects as ``key=value`` pairs.

    Args:
        value: The cell value.

    Returns:
        The text to show.
    """
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, str) and value.startswith("{"):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        return ", ".join(f"{k}={_format(v)}" for k, v in parsed.items()) or "-"
    return str(value)


def _table(rows: Sequence[dict[str, object]], columns: Sequence[str]) -> "Component":
    """Build a semantic HTML table in a scrollable box.

    The styling lives in ``assets/dashboard.css``: a sticky header, zebra rows, fixed-width
    digits so numeric columns line up, and ``content-visibility`` so offscreen rows of a long
    table are not laid out.

    Args:
        rows: The table rows, keyed by column name.
        columns: The columns to show, in order.

    Returns:
        The table component; a short message when there are no rows.
    """
    from dash import html

    if not rows:
        return html.P("No rows", className="vl-empty")
    head = html.Tr([html.Th(name, scope="col") for name in columns])
    body = [html.Tr([html.Td(_format(row[name])) for name in columns]) for row in rows]
    table = html.Table([html.Thead(head), html.Tbody(body)], className="vl-table")
    return html.Div(table, className="vl-scroll")


def _live(region_id: str) -> "Component":
    """Build an empty container whose later updates a screen reader announces.

    Args:
        region_id: The component id a callback writes into.

    Returns:
        A ``div`` marked ``aria-live="polite"``.
    """
    from dash import html

    # Dash passes any aria-* attribute through; its type stubs list only the documented props.
    return html.Div(id=region_id, **{"aria-live": "polite"})  # type: ignore[arg-type]


def _readout(readings: Sequence[tuple[str, str]]) -> "Component":
    """Build the strip of readings that sits above the chosen drive's figures.

    Args:
        readings: ``(label, value)`` pairs from ``drive_stats``.

    Returns:
        The strip, or a line of guidance when no drive is chosen.
    """
    from dash import html

    if not readings:
        return html.P(NO_DRIVE_NOTICE, className="vl-empty")
    cells = [
        html.Div(
            [
                html.Span(label, className="vl-read-label"),
                html.Span(value, className="vl-read-value"),
            ],
            className="vl-read",
        )
        for label, value in readings
    ]
    return html.Div(cells, className="vl-readout")


def _card(heading: str, *content: "Component") -> "Component":
    """Group a section under a heading in a bordered card.

    Args:
        heading: The section heading (rendered as an ``h2`` under the page's ``h1``).
        *content: The section's components, in order.

    Returns:
        The card component.
    """
    from dash import html

    return html.Div([html.H2(heading), *content], className="vl-card")


def _controls(options: list[dict[str, str]]) -> list["Component"]:
    """Build the drive dropdown and the event-type filter, with the empty-warehouse notice.

    Args:
        options: The drive dropdown options.

    Returns:
        The components, in page order.
    """
    from dash import dcc, html

    kinds = [e.value for e in EventType]
    notice = [] if options else [html.P(NO_DATA_NOTICE, className="vl-notice")]
    first = options[0]["value"] if options else None
    return [
        *notice,
        html.Label("Drive", htmlFor="drive", className="vl-field"),
        dcc.Dropdown(id="drive", options=options, value=first),
        html.Label("Event Types", htmlFor="kinds", className="vl-field"),
        dcc.Dropdown(id="kinds", options=kinds, value=kinds, multi=True),
        html.P("Filters both event tables below.", className="vl-empty"),
    ]


def _graph(graph_id: str, height: int) -> "Component":
    """Build a figure that fills a fixed height and follows the card's width.

    Args:
        graph_id: The component id a callback writes the figure into.
        height: The drawing height in pixels (the figure itself sets none).

    Returns:
        The graph component.
    """
    from dash import dcc

    return dcc.Graph(id=graph_id, style={"height": f"{height}px"}, responsive=True)


def _explorer_tab(options: list[dict[str, str]]) -> "Component":
    """Build the content of the "Scenario Explorer" tab.

    The controls stay in a sticky column on the left; the drive's readings and its two
    figures fill the wide column; the event tables run the full width underneath.

    Args:
        options: The drive dropdown options.

    Returns:
        The tab content.
    """
    from dash import html

    # The trajectory is the tallest figure: its axes are equally scaled, so it needs the room.
    drive = html.Div(
        [
            _live("readout"),
            _card("Trajectory", _graph("trajectory", 420)),
            _card("Speed & Acceleration", _graph("speed", 300)),
        ],
        className="vl-stack",
    )
    tables = [
        _card("Events of the Chosen Drive", _live("drive-events")),
        _card("Events of All Drives", _live("all-events")),
    ]
    side = html.Div(_card("Choose a Drive", *_controls(options)), className="vl-side")
    full = [html.Div(table, className="vl-full") for table in tables]
    return html.Div([side, drive, *full], className="vl-explorer")


def _quality_tab() -> "Component":
    """Build the content of the "Data Quality" tab.

    Reads top to bottom as the question a reader asks: how much was unusable, which rules
    fired, which drives they fired on, and what each code means.

    Returns:
        The tab content.
    """
    from dash import html

    terms: list["Component"] = []
    for code in ErrorCode:
        terms += [html.Dt(code.value), html.Dd(code.description)]
    # see _live: aria-* is passed through but is not in the stubs
    count = html.Strong(id="rejects", className="vl-stat", **{"aria-live": "polite"})  # type: ignore[arg-type]
    note = html.P("Delivered by Kafka, but unparsable (VL-005).", className="vl-empty")
    stat = html.Div(_card("Rejected Messages", count, note), className="vl-side")
    chart = _card("Frames per Error Code", _graph("errors", 280))
    table = _card("Error Counts per Drive & Code", html.Div(id="errors-per-drive"))
    codes = _card("Error Codes", html.Dl(terms, className="vl-legend"))
    full = [html.Div(part, className="vl-full") for part in (table, codes)]
    return html.Div([stat, chart, *full], className="vl-quality")


def _tab(label: str, value: str, content: "Component") -> "Component":
    """Build one tab: a quiet label when idle, a panel joined to its content when chosen.

    Args:
        label: The tab label.
        value: The tab value.
        content: The tab content.

    Returns:
        The tab component.
    """
    from dash import dcc

    shared = {"border": "none", "padding": "9px 16px", "fontSize": "13px", "lineHeight": "1.4"}
    shared |= {"backgroundColor": "transparent", "borderRadius": "6px 6px 0 0"}
    idle = {**shared, "color": "#5d6b75"}
    chosen = {**shared, "backgroundColor": "#ffffff", "color": "#111a21", "fontWeight": "600"}
    # The 2px edge comes out of the padding, so choosing a tab does not move its label.
    chosen |= {"borderTop": "2px solid #0f7b78", "paddingTop": "7px"}
    return dcc.Tab(label=label, value=value, children=content, style=idle, selected_style=chosen)


def _layout(options: list[dict[str, str]]) -> "Component":
    """Build the page: the header band, then the two tabs on the page measure.

    Args:
        options: The drive dropdown options.

    Returns:
        The page layout.
    """
    from dash import dcc, html

    tabs = dcc.Tabs(
        id="tabs",
        value="explorer",
        children=[
            _tab("Scenario Explorer", "explorer", _explorer_tab(options)),
            _tab("Data Quality", "quality", _quality_tab()),
        ],
    )
    title = html.Div([html.H1("VL Drives"), html.P(SUBTITLE)], className="vl-wrap")
    band = html.Header(title, className="vl-band")
    return html.Div([band, html.Div(tabs, className="vl-wrap")])


def _register_drive(app: "dash.Dash", wh: Warehouse) -> None:
    """Connect the drive dropdown and the event filter to the drive figures and table.

    Args:
        app: The app.
        wh: The warehouse the callbacks query.
    """
    from dash import Input, Output
    from dash.development.base_component import Component

    @app.callback(
        Output("trajectory", "figure"),
        Output("speed", "figure"),
        Output("drive-events", "children"),
        Output("readout", "children"),
        Input("drive", "value"),
        Input("kinds", "value"),
    )
    def on_drive(
        drive_id: str | None, kinds: list[str] | None
    ) -> tuple[go.Figure, go.Figure, Component, Component]:
        trajectory, speed, rows = drive_view(wh, drive_id)
        shown = [r for r in rows if r["event_type"] in (kinds or [])]
        readout = _readout(drive_stats(wh, drive_id))
        return trajectory, speed, _table(shown, EVENT_COLUMNS), readout


def _register_cross_drive(app: "dash.Dash", wh: Warehouse) -> None:
    """Connect the event filter to the cross-drive events table.

    Args:
        app: The app.
        wh: The warehouse the callbacks query.
    """
    from dash import Input, Output
    from dash.development.base_component import Component

    @app.callback(Output("all-events", "children"), Input("kinds", "value"))
    def on_kinds(kinds: list[str] | None) -> Component:
        return _table(_filtered_events(wh, kinds or []), ["drive_id", *EVENT_COLUMNS])


def _register_quality(app: "dash.Dash", wh: Warehouse) -> None:
    """Connect the tab switch to the data-quality content.

    Args:
        app: The app.
        wh: The warehouse the callbacks query.
    """
    from dash import Input, Output
    from dash.development.base_component import Component

    @app.callback(
        Output("errors", "figure"),
        Output("rejects", "children"),
        Output("errors-per-drive", "children"),
        Input("tabs", "value"),
    )
    def on_quality(_tab: str | None) -> tuple[go.Figure, int, Component]:
        figure, rejects = quality_view(wh)
        rows = _records(wh.query(ERRORS_PER_DRIVE_SQL))
        return figure, rejects, _table(rows, ["drive_id", "error_code", "n_frames"])


def create_app(wh: Warehouse) -> "dash.Dash":
    """Build the Dash app with the "Scenario Explorer" and "Data Quality" tabs.

    Args:
        wh: The warehouse the callbacks query.

    Returns:
        The configured app, not yet running.
    """
    import dash

    app = dash.Dash(__name__, title="VL data product", meta_tags=META_TAGS)
    app.layout = _layout(drive_options(wh))
    for register in (_register_drive, _register_cross_drive, _register_quality):
        register(app, wh)
    return app


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dashboard server: ``python -m vl.dashboard.app --warehouse DIR``.

    Args:
        argv: Command-line arguments (``--warehouse DIR``, ``--port N``, ``--host H``);
            None means ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(prog="vl.dashboard.app", description="VL dashboard server.")
    parser.add_argument("--warehouse", default="data/warehouse", help="warehouse directory")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    create_app(Warehouse(args.warehouse)).run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
