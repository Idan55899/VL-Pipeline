"""Review round 1, item 4: CLI input errors exit 2 with a one-line message; empty source fails."""

import json
from pathlib import Path

import pytest

from sim_helpers import make_frames
from vl.validate import main
from vl.validation import CHECK_SOURCE_NOT_EMPTY


def good_source(tmp_path: Path) -> Path:
    """Write a valid 100-frame source."""
    path = tmp_path / "frames.jsonl"
    path.write_text("".join(json.dumps(m) + "\n" for m in make_frames(100)))
    return path


def run(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """Run ``main`` and return the exit code, stdout and stderr."""
    code = main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def assert_one_line_error(case: str, code: int, out: str, err: str) -> None:
    """Assert the run exited 2 with a single non-empty line on stderr and no report."""
    assert code == 2, f"{case}: exit code {code}"
    assert len(err.strip().splitlines()) == 1, f"{case}: stderr {err!r}"
    assert err.strip(), case
    assert out == "", f"{case}: stdout {out!r}"
    assert "Traceback" not in err, case


def test_unreadable_source_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    undecodable = tmp_path / "undecodable.jsonl"
    undecodable.write_bytes(b"\xff\xfe\x00\x80 not utf-8\n")
    sources = {
        "missing source": tmp_path / "nope.jsonl",
        "directory source": tmp_path,
        "undecodable source": undecodable,
    }
    for case, path in sources.items():
        assert_one_line_error(case, *run(["--source", str(path)], capsys))


def test_bad_labels_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad_labels = {
        "not json": "{not json",
        "not an object": "[1, 2, 3]",
        "no labels key": json.dumps({"drives": []}),
        "labels not a list": json.dumps({"labels": "nope"}),
        "label not an object": json.dumps({"labels": [1, 2]}),
        "label without expected_code": json.dumps({"labels": [{"drive_id": "d"}]}),
        "label with bad frame_idx": json.dumps(
            {"labels": [{"drive_id": "d", "frame_idx": "x", "expected_code": None}]}
        ),
    }
    source = good_source(tmp_path)
    labels = tmp_path / "labels.json"
    for case, content in bad_labels.items():
        labels.write_text(content)
        args = ["--source", str(source), "--labels", str(labels)]
        assert_one_line_error(case, *run(args, capsys))
    args = ["--source", str(source), "--labels", str(tmp_path / "none.json")]
    assert_one_line_error("missing labels file", *run(args, capsys))


def test_argparse_errors_stay_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2


def test_empty_source_fails_the_named_check_and_blank_lines_count_as_vl005(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "frames.jsonl"
    expected = {"": f"FAIL {CHECK_SOURCE_NOT_EMPTY}", "\n  \n": "VL-005=2"}  # blank = malformed
    for content, needle in expected.items():
        path.write_text(content)
        code, out, err = run(["--source", str(path)], capsys)
        assert (code, err) == (1, ""), content
        assert needle in out, content


def test_non_empty_source_passes_the_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(["--source", str(good_source(tmp_path))], capsys)
    assert code == 0, out
    assert f"PASS {CHECK_SOURCE_NOT_EMPTY}" in out


def test_invalid_env_value_exits_2_with_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for value in ("abc", "2"):  # not a number, and out of the (0, 1) range
        monkeypatch.setenv("VL_MAX_FLAG_RATE", value)
        code, out, err = run(["--source", str(good_source(tmp_path))], capsys)
        assert_one_line_error(f"VL_MAX_FLAG_RATE={value}", code, out, err)
        assert err.startswith("error:") and "max_flag_rate" in err.lower(), err
