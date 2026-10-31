"""Load tests of the dashboard read path: concurrent queries and a drive with many events."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from pipeline_helpers import frames_table
from vl.dashboard.app import drive_view
from vl.dashboard.figures import speed_fig
from vl.schemas import EVENTS_COLUMNS, EVENTS_DTYPES
from vl.warehouse import Warehouse


def make_events(n: int) -> pd.DataFrame:
    """Build ``n`` one-frame events spread over a drive."""
    rows = [("d", "STOP", i, i + 1, i / 16, (i + 1) / 16, "{}") for i in range(n)]
    return pd.DataFrame(rows, columns=EVENTS_COLUMNS).astype(EVENTS_DTYPES)


def test_concurrent_queries_do_not_collide(tmp_path: Path) -> None:
    frames = frames_table(50)
    for name, table in (("frames", frames), ("events", make_events(3))):
        (tmp_path / name).mkdir()
        table.to_parquet(tmp_path / name / "part.parquet")
    wh = Warehouse(str(tmp_path))

    def work(_: int) -> tuple[int, int]:
        trajectory, _speed, rows = drive_view(wh, "d")
        return len(trajectory.data[0].x), len(rows)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(work, range(64)))
    assert results == [(50, 3)] * 64


def test_speed_fig_with_many_events_is_fast() -> None:
    frames = frames_table(2000)
    start = time.perf_counter()
    fig = speed_fig(frames, make_events(200))
    assert len(fig.layout.shapes) == 200
    assert time.perf_counter() - start < 5.0
