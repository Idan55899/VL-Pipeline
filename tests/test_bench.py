"""Tests for the benchmark (`vl.bench`): it runs quickly and its numbers have the right shape."""

import pytest

from vl.bench import main, run_bench


def test_quick_bench_measures_formats_and_transform(capsys: pytest.CaptureFixture[str]) -> None:
    rows = {row.name: row for row in run_bench(quick=True)}
    for points in (200, 10_000):
        json_size = rows[f"json bytes/frame @{points} pts"].value
        assert rows[f"binary bytes/frame @{points} pts"].value < json_size
        assert rows[f"binary decode+centroid @{points} pts"].value >= 0
    assert rows["derive_frames rows/s @1000-frame drive"].value > 0
    codec = rows["codec bytes/drive @200-frame drive"].value
    assert codec * 10 < rows["json cloud bytes/drive @200-frame drive"].value
    assert main(["--quick"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| measurement | value | unit |")
