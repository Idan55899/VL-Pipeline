"""Tests for the validation report (validate_frames, validate_realism) and the validate CLI."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fixtures.loader import load_golden, raw_dataframe
from pipeline_helpers import frames_table, raw_from_frames, with_codes
from sim_helpers import make_frames
from vl.etl.features import derive_frames
from vl.mock.drive_sim import simulate_fleet
from vl.mock.generate import write_fleet_jsonl
from vl.validate import main
from vl.validation import (
    CHECK_ACCEL_P99,
    CHECK_CODE_COUNTS,
    CHECK_DTYPES,
    CHECK_FALSE_POSITIVES,
    CHECK_FINITE,
    CHECK_FLAG_RATE,
    CHECK_KNOWN_PROFILE,
    CHECK_KNOWN_VEHICLE_KIND,
    CHECK_LABELED_FAULTS,
    CHECK_MAX_SPEED,
    CHECK_NO_NULLS,
    CHECK_REQUIRED_COLUMNS,
    CHECK_TS_ORDER,
    CHECK_YAW_RATE_P99,
    CheckResult,
    ValidationReport,
    validate_frames,
    validate_realism,
)

FRAMES_CHECKS = [
    CHECK_REQUIRED_COLUMNS,
    CHECK_DTYPES,
    CHECK_NO_NULLS,
    CHECK_FINITE,
    CHECK_TS_ORDER,
    CHECK_CODE_COUNTS,
    CHECK_FLAG_RATE,
]
REALISM_CHECKS = [
    CHECK_KNOWN_PROFILE,
    CHECK_KNOWN_VEHICLE_KIND,
    CHECK_MAX_SPEED,
    CHECK_ACCEL_P99,
    CHECK_YAW_RATE_P99,
    CHECK_FLAG_RATE,
]
LABELS_META = {
    "drive_id": "d",
    "vehicle_id": "veh-1",
    "vehicle_kind": "car",
    "profile": "scripted",
    "drive_start_utc": "2026-01-01T00:00:00+00:00",
    "n_frames": 100,
    "seed": 0,
}


# --- helpers ------------------------------------------------------------------------------


def failed(report: ValidationReport) -> set[str]:
    """Return the names of the failed checks."""
    return {check.name for check in report.checks if not check.passed}


def check_named(report: ValidationReport, name: str) -> CheckResult:
    """Return the check with the given name (asserting it exists once)."""
    matches = [check for check in report.checks if check.name == name]
    assert len(matches) == 1, f"expected one check named {name!r}, got {report.checks}"
    return matches[0]


# --- validate_frames ----------------------------------------------------------------------


def test_valid_frames_pass_every_check() -> None:
    golden = derive_frames(raw_dataframe(load_golden("g01_constant_speed")))
    report = validate_frames(golden)
    assert report.passed, report.to_text()
    report = validate_frames(frames_table(100))
    assert report.passed, report.to_text()
    assert [check.name for check in report.checks] == FRAMES_CHECKS


def test_structural_violations_fail_their_named_check() -> None:
    nan_x = frames_table(100)
    nan_x.loc[40, "x"] = np.nan
    infinite = frames_table(100)
    infinite.loc[10, "speed"] = np.inf
    wrong_dtype = frames_table(100)
    wrong_dtype["frame_idx"] = wrong_dtype["frame_idx"].astype(float)
    decreasing_ts = frames_table(100)
    decreasing_ts.loc[50, "ts"] = decreasing_ts.loc[49, "ts"] - 0.001
    cases = [
        (
            "missing column",
            frames_table(100).drop(columns=["accel"]),
            CHECK_REQUIRED_COLUMNS,
            "'accel'",
        ),
        ("null position", nan_x, CHECK_NO_NULLS, "'x'"),
        ("infinite value", infinite, CHECK_FINITE, ""),
        ("wrong dtype", wrong_dtype, CHECK_DTYPES, "'frame_idx'"),
        ("decreasing ts", decreasing_ts, CHECK_TS_ORDER, ""),
    ]
    for case, frames, name, detail in cases:
        report = validate_frames(frames)
        check = check_named(report, name)
        assert not check.passed, case
        assert detail in check.detail, f"{case}: {check.detail}"
        assert not report.passed, case


def test_tolerated_inputs_pass() -> None:
    null_kinematics = frames_table(100)
    null_kinematics.loc[0, ["vx", "vy", "speed", "accel", "yaw_rate"]] = np.nan
    assert validate_frames(null_kinematics).passed, "null kinematics are allowed"

    first, second = frames_table(50), frames_table(50).assign(drive_id="e")
    both = pd.concat([first, second], ignore_index=True)
    assert validate_frames(both).passed, "ts order is checked per drive"

    empty = validate_frames(frames_table(0))
    assert empty.passed, empty.to_text()
    assert check_named(empty, CHECK_FLAG_RATE).passed


def test_flagged_vl004_rows_are_exempt_from_the_ts_order_check() -> None:
    backwards = frames_table(100)
    backwards.loc[50, "ts"] = backwards.loc[49, "ts"] - 0.001
    report = validate_frames(with_codes(backwards, [50], "VL-004"))
    assert check_named(report, CHECK_TS_ORDER).passed, "VL-004 rows are exempt from the ts check"
    assert report.passed, report.to_text()

    golden = derive_frames(raw_dataframe(load_golden("g08_ts_backwards")))
    assert check_named(validate_frames(golden), CHECK_TS_ORDER).passed, "derived g08"


def test_flag_rate_limit_is_inclusive_and_follows_max_flag_rate() -> None:
    for flagged, expected_passed in [(5, True), (6, False)]:
        frames = with_codes(frames_table(100), list(range(flagged)), "VL-006")
        report = validate_frames(frames)
        assert check_named(report, CHECK_FLAG_RATE).passed == expected_passed, flagged
    twenty_percent = with_codes(frames_table(100), list(range(0, 100, 5)), "VL-001")
    assert failed(validate_frames(twenty_percent)) == {CHECK_FLAG_RATE}
    ten_percent = with_codes(frames_table(100), list(range(10)), "VL-001")
    assert CHECK_FLAG_RATE in failed(validate_frames(ten_percent))
    assert validate_frames(ten_percent, max_flag_rate=0.2).passed


def test_error_code_counts_are_reported() -> None:
    frames = with_codes(frames_table(100), [3, 4], "VL-001")
    frames = with_codes(frames, [50], "VL-002")
    check = check_named(validate_frames(frames), CHECK_CODE_COUNTS)
    assert check.passed
    assert "VL-001" in check.detail
    assert "VL-002" in check.detail


# --- ValidationReport ---------------------------------------------------------------------


def test_report_text_lists_every_check() -> None:
    report = validate_frames(frames_table(100).drop(columns=["accel"]))
    text = report.to_text()
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        assert f"{status} {check.name}: {check.detail}" in text


def test_report_passed_requires_every_check() -> None:
    ok, bad = CheckResult("a", True, "fine"), CheckResult("b", False, "broken")
    assert ValidationReport((ok,)).passed
    assert not ValidationReport((ok, bad)).passed


# --- validate_realism ---------------------------------------------------------------------


def test_realism_passes_on_simulated_fleet() -> None:
    for drive in simulate_fleet(4, seed=0, n_frames=2000):
        frames = derive_frames(raw_from_frames(drive.frames))
        report = validate_realism(frames, drive.meta.profile, drive.meta.vehicle_kind)
        assert report.passed, f"{drive.meta.drive_id}: {report.to_text()}"


def test_realism_passes_and_names_its_checks_on_constructed_baseline() -> None:
    report = validate_realism(frames_table(200), "city", "car")
    assert report.passed, report.to_text()
    assert [check.name for check in report.checks] == REALISM_CHECKS


def test_realism_fails_exactly_the_exceeded_limit() -> None:
    every_tenth = np.zeros(200)
    every_tenth[::10] = 1.0  # 10% of the frames, so the 99th percentile is the extreme value
    few_outliers = np.zeros(1000)
    few_outliers[:5] = 50.0  # 0.5% of the frames: below the 99th percentile
    cases = [
        ("speed", frames_table(200, speed=80.0, vx=80.0), {CHECK_MAX_SPEED}),
        ("accel p99", frames_table(200, accel=50.0 * every_tenth), {CHECK_ACCEL_P99}),
        ("yaw p99", frames_table(200, yaw_rate=2.0 * every_tenth), {CHECK_YAW_RATE_P99}),
        (
            "flag rate",
            with_codes(frames_table(200), list(range(0, 200, 10)), "VL-001"),
            {CHECK_FLAG_RATE},
        ),
        ("outliers below p99 are ignored", frames_table(1000, accel=few_outliers), set()),
    ]
    for case, frames, want in cases:
        assert failed(validate_realism(frames, "city", "car")) == want, case


def test_realism_unknown_profile_or_vehicle_kind_fails_instead_of_raising() -> None:
    physics = (CHECK_MAX_SPEED, CHECK_ACCEL_P99, CHECK_YAW_RATE_P99)
    report = validate_realism(frames_table(200), "no-such-profile", "car")
    assert not check_named(report, CHECK_KNOWN_PROFILE).passed
    assert set(physics) <= failed(report)
    for name in physics:
        assert "unknown profile" in check_named(report, name).detail
    report = validate_realism(frames_table(200), "city", "hovercraft")
    assert not check_named(report, CHECK_KNOWN_VEHICLE_KIND).passed
    assert check_named(report, CHECK_KNOWN_PROFILE).passed


def test_realism_limits_depend_on_the_profile() -> None:
    frames = frames_table(200, speed=10.0)  # fine for city (17 m/s), too fast for parking (4 m/s)
    assert validate_realism(frames, "city", "car").passed
    assert CHECK_MAX_SPEED in failed(validate_realism(frames, "parking", "car"))


# --- CLI ----------------------------------------------------------------------------------


def write_jsonl(path: Path, messages: list[dict[str, Any]]) -> Path:
    """Write messages as JSON lines and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(message) + "\n" for message in messages))
    return path


def write_labels(path: Path, labels: list[dict[str, Any]]) -> Path:
    """Write a ``labels.json`` in the generator's format and return the path."""
    path.write_text(json.dumps({"drives": [LABELS_META], "labels": labels}))
    return path


def sparse_label(frame_idx: int) -> dict[str, Any]:
    """Return the label of a sparse-cloud fault expecting VL-006."""
    return {"drive_id": "d", "frame_idx": frame_idx, "kind": "sparse", "expected_code": "VL-006"}


def with_sparse_cloud(message: dict[str, Any]) -> dict[str, Any]:
    """Keep 10 central points of a hand-made frame (same centroid, so only VL-006 fires)."""
    return {**message, "points": message["points"][25:35]}


def sparse_at(messages: list[dict[str, Any]], frame_ids: list[int]) -> list[dict[str, Any]]:
    """Return the messages with a sparse cloud at the given frame indices."""
    return [with_sparse_cloud(m) if m["frame_idx"] in frame_ids else m for m in messages]


def corrupt_every_fifth_line(source: Path, target: Path) -> Path:
    """Copy a ``frames.jsonl`` truncating the cloud of every fifth message to 10 points."""
    lines = [json.loads(line) for line in source.read_text().splitlines()]
    for i in range(0, len(lines), 5):
        lines[i]["points"] = lines[i]["points"][:10]
    return write_jsonl(target, lines)


def run_cli(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    """Run ``main`` and return its exit code and captured stdout."""
    code = main(args)
    return code, capsys.readouterr().out


def test_cli_exit_codes_and_report_on_hand_made_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = write_jsonl(tmp_path / "good.jsonl", make_frames(100))
    bad = corrupt_every_fifth_line(good, tmp_path / "bad.jsonl")
    code, out = run_cli(["--source", str(good)], capsys)
    assert code == 0, out
    for name in FRAMES_CHECKS:
        assert name in out
    code, out = run_cli(["--source", str(bad)], capsys)
    assert code == 1
    assert CHECK_FLAG_RATE in out


def test_cli_reports_collapsed_duplicates_and_lost_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Frames 10-12 and 50 are missing (4 lost, 2 gaps); frame 70 is sent 3 times (2 collapsed)."""
    kept = [m for m in make_frames(100) if m["frame_idx"] not in (10, 11, 12, 50)]
    extra = [m for m in kept if m["frame_idx"] == 70] * 2
    _, out = run_cli(["--source", str(write_jsonl(tmp_path / "f.jsonl", kept + extra))], capsys)
    assert "PASS duplicates_collapsed: 2 row(s)" in out.splitlines()
    assert "PASS frames_lost: 4 frame(s) in 2 gap(s)" in out.splitlines()


def bad_lines(kind: str, count: int) -> list[str]:
    """Return ``count`` invalid message lines of the given kind."""
    if kind == "not_json":
        return ["{this is not json"] * count
    if kind == "deep_json":  # deeper than the Python recursion limit
        return ["[" * 100000 + "]" * 100000] * count
    if kind == "huge_ts":  # an integer too large for a float
        return [json.dumps({**make_frames(1)[0], "frame_idx": 5000, "ts": 10**400})] * count
    garbage = {**make_frames(1)[0], "frame_idx": 5000, "points": "garbage"}
    return [json.dumps(garbage)] * count


def write_with_bad_lines(path: Path, good: int, kind: str, bad: int) -> Path:
    """Write ``good`` valid hand-made frames with ``bad`` invalid lines mixed in."""
    lines = [json.dumps(m) for m in make_frames(good)]
    for i, line in enumerate(bad_lines(kind, bad)):
        lines.insert(3 + i * 2, line)
    path.write_text("".join(line + "\n" for line in lines))
    return path


def test_cli_counts_malformed_lines_as_vl005(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One malformed line in 100 passes, ten in 100 fail, for both kinds of bad line."""
    for kind in ("not_json", "garbage_points", "deep_json", "huge_ts"):
        for good, bad, exit_code in [(99, 1, 0), (90, 10, 1)]:
            source = write_with_bad_lines(tmp_path / "frames.jsonl", good, kind, bad)
            code, out = run_cli(["--source", str(source)], capsys)
            assert code == exit_code, f"{kind} {bad}/100: {out}"
            assert "VL-005" in out, f"{kind} {bad}/100"


def statuses(out: str) -> dict[str, str]:
    """Map each check name to PASS/FAIL from report lines ``<PASS|FAIL> <name>: <detail>``."""
    found = {}
    for line in out.splitlines():
        status, _, rest = line.partition(" ")
        if status in ("PASS", "FAIL"):
            found[rest.split(":", 1)[0]] = status
    return found


def run_with_labels(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    lines: list[str],
    labels: list[dict[str, Any]],
) -> tuple[int, dict[str, str]]:
    """Write the JSONL lines and labels, run the CLI with ``--labels``, return code and statuses."""
    source = tmp_path / "frames.jsonl"
    source.write_text("".join(line + "\n" for line in lines))
    labels_path = write_labels(tmp_path / "labels.json", labels)
    code, out = run_cli(["--source", str(source), "--labels", str(labels_path)], capsys)
    return code, statuses(out)


def dumps(messages: list[dict[str, Any]]) -> list[str]:
    """Serialize messages to JSON lines."""
    return [json.dumps(message) for message in messages]


def label(kind: str, frame_idx: int, code: str | None) -> dict[str, Any]:
    """Return a fault label of the generator's format."""
    return {"drive_id": "d", "frame_idx": frame_idx, "kind": kind, "expected_code": code}


def shifted_by_5m(messages: list[dict[str, Any]], first_idx: int) -> list[dict[str, Any]]:
    """Shift the cloud x by 5 m for every frame at or after ``first_idx``."""
    shift = [500, 0, 0]
    return [
        {**m, "points": [[a + b for a, b in zip(p, shift)] for p in m["points"]]}
        if m["frame_idx"] >= first_idx
        else m
        for m in messages
    ]


def detected_fault_case(kind: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return hand-made messages with one fault of ``kind`` and its label."""
    frames = make_frames(100)
    if kind == "sparse":
        return sparse_at(frames, [50]), label("sparse", 50, "VL-006")
    if kind == "drop":
        return [m for m in frames if m["frame_idx"] != 50], label("drop", 51, "VL-001")
    if kind == "jump":
        return shifted_by_5m(frames, 50), label("jump", 50, "VL-002")
    duplicated = [m for f in frames for m in ([f, f] if f["frame_idx"] == 50 else [f])]
    return duplicated, label("duplicate", 50, "VL-004")


def test_labels_check_passes_when_the_fault_is_detected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for kind in ("sparse", "drop", "jump", "duplicate"):
        messages, fault_label = detected_fault_case(kind)
        code, found = run_with_labels(tmp_path, capsys, dumps(messages), [fault_label])
        assert found[CHECK_LABELED_FAULTS] == "PASS", kind
        assert found[CHECK_FALSE_POSITIVES] == "PASS", kind
        assert code == 0, kind


def test_labels_check_fails_for_a_missed_fault_and_for_a_wrong_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Missed: the label expects VL-006 on frame 50 but the frame has a full cloud.
    # Wrong code: frame 50 gets VL-006 but the label says VL-002.
    sparse_messages, _ = detected_fault_case("sparse")
    cases = [
        ("missed fault", dumps(make_frames(100)), [sparse_label(50)]),
        ("wrong code", dumps(sparse_messages), [label("jump", 50, "VL-002")]),
    ]
    for case, lines, labels in cases:
        code, found = run_with_labels(tmp_path, capsys, lines, labels)
        assert found[CHECK_LABELED_FAULTS] == "FAIL", case
        assert code == 1, case


def test_labels_check_ignores_labels_without_an_expected_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A malformed message has expected_code null: it never reaches frames, so nothing to detect.
    lines = dumps(make_frames(99)) + bad_lines("garbage_points", 1)
    code, found = run_with_labels(tmp_path, capsys, lines, [label("malformed", 5000, None)])
    assert found[CHECK_LABELED_FAULTS] == "PASS"
    assert code == 0


def test_false_positive_rate_check_fails_above_one_percent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Two unlabeled flagged frames out of 100 (2%): over the 1% limit, under the 5% flag rate.
    lines = dumps(sparse_at(make_frames(100), [30, 60]))
    code, found = run_with_labels(tmp_path, capsys, lines, [])
    assert found[CHECK_FALSE_POSITIVES] == "FAIL"
    assert found[CHECK_FLAG_RATE] == "PASS"
    assert code == 1


# --- CLI --realism --------------------------------------------------------------------------


def realism_fleet(directory: Path) -> tuple[Path, Path, Path]:
    """Write a 3-drive fleet as frames.jsonl, a raw_frames Parquet table and labels.json."""
    drives = simulate_fleet(3, seed=0, n_frames=1000)
    write_fleet_jsonl(drives, str(directory))
    table = directory / "raw_frames"
    raw = pd.concat([raw_from_frames(drive.frames) for drive in drives], ignore_index=True)
    raw.to_parquet(table, partition_cols=["drive_id"])
    return directory / "frames.jsonl", table, directory / "labels.json"


def realism_names(out: str, drive_id: str) -> list[str]:
    """Return the names of the ``realism[<drive_id>].`` lines of a report, in order."""
    return [name for name in statuses(out) if name.startswith(f"realism[{drive_id}].")]


def test_cli_realism_appends_per_drive_checks_for_both_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jsonl, table, labels = realism_fleet(tmp_path)
    for source in (jsonl, table):
        base = ["--source", str(source), "--labels", str(labels)]
        _, plain = run_cli(base, capsys)
        code, out = run_cli([*base, "--realism"], capsys)
        assert code == 0, out
        assert "FAIL" not in out
        for drive_id in ("drive-001", "drive-002", "drive-003"):
            want = [f"realism[{drive_id}].{name}" for name in REALISM_CHECKS]
            assert realism_names(out, drive_id) == want, source
        assert out.splitlines()[: len(plain.splitlines())] == plain.splitlines(), "appended only"


def test_cli_realism_input_errors_exit_2_with_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jsonl, _, labels = realism_fleet(tmp_path)
    payload = json.loads(labels.read_text())
    payload["drives"] = payload["drives"][1:]  # drive-001 has data but no metadata
    short = tmp_path / "short.json"
    short.write_text(json.dumps(payload))
    needs = "error: --realism needs --labels (the profile and vehicle kind come from labels.json)"
    with_short = ["--source", str(jsonl), "--labels", str(short), "--realism"]
    cases = [
        ("no labels", ["--source", str(jsonl), "--realism"], needs),
        ("unknown drive", with_short, "drive-001"),
    ]
    for case, args, want in cases:
        code = main(args)
        captured = capsys.readouterr()
        lines = captured.err.splitlines()
        assert code == 2, case
        assert captured.out == "", case
        assert len(lines) == 1 and lines[0].startswith("error:") and want in lines[0], case


def test_cli_realism_fails_a_drive_whose_data_breaks_its_profile_limits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jsonl, _, labels = realism_fleet(tmp_path)
    hostile = "drive-001\n\ud800"  # a newline and a lone surrogate must not break the report
    for path in (jsonl, labels):
        path.write_text(path.read_text().replace('"drive-001"', json.dumps(hostile)))
    payload = json.loads(labels.read_text())
    payload["drives"][0]["profile"] = "parking"  # drive-001 is a highway drive (over 30 m/s)
    labels.write_text(json.dumps(payload))
    code, out = run_cli(["--source", str(jsonl), "--labels", str(labels), "--realism"], capsys)
    found = statuses(out)
    assert code == 1, out
    assert found[f"realism[drive-001\\n\\ud800].{CHECK_MAX_SPEED}"] == "FAIL"
    others = {name: status for name, status in found.items() if "drive-001" not in name}
    assert "FAIL" not in {status for name, status in others.items() if name.startswith("realism")}
