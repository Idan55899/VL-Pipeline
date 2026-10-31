"""Phase 1 sanity test: one fast end-to-end run of the real code on the mock data.

Role in the process (mock data -> unit tests -> implement -> sanity -> next phase): after the
unit tests are green, this module runs every layer built in Phase 1 together, with no stubs
and no mocks of project code:

* simulator -> ``frames.jsonl`` / ``labels.json`` -> validator CLI (``vl.validate.main``);
* simulator -> raw rows -> ``derive_frames`` -> ``detect_events`` -> ``summarize_drive``.

It asserts the key outputs against the spec (scripted scenario, label recall, false-positive
rate, realism limits), never against values obtained by running the code under test. It is
selectable alone with ``pytest -m sanity`` (``make sanity``) and also runs in the default suite.
"""

import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pipeline_helpers import raw_from_frames
from vl.config import DEFAULT
from vl.errors import ErrorCode
from vl.etl.features import derive_frames, detect_events, summarize_drive
from vl.mock.drive_sim import SimConfig, SimulatedDrive, simulate_drive, simulate_fleet
from vl.mock.generate import write_fleet_jsonl
from vl.validate import main as validate_main
from vl.validation import validate_realism

pytestmark = pytest.mark.sanity

N_DRIVES = 5
N_FRAMES = 1000
# Seed 0 with 5 drives of 1000 frames already contains every fault kind below (checked by
# test_every_labeled_fault_is_detected_with_its_code), so no larger fleet is needed.
FLEET_SEED = 0
FAULT_KINDS = {"drop", "jump", "duplicate", "sparse", "malformed", "gap_after_malformed"}
SCRIPTED_DISTANCE_M = 60 + 180 + 20 + 25 + 15 + 40 + 150 + 56.25  # hand-derived from the contract
HARD_BRAKE_FRAME = 320  # t = 20 s at 16 fps
SHARP_TURN_FRAME = 432  # t = 27 s at 16 fps
TOLERANCE_FRAMES = 10
KEY = ["drive_id", "frame_idx"]


class Fleet(NamedTuple):
    """The fleet built once for the module.

    Attributes:
        drives: The simulated drives.
        out_dir: Directory holding ``frames.jsonl`` and ``labels.json``.
        frames: Derived frames per ``drive_id``.
        labels: All fault labels of the fleet as a table.
    """

    drives: list[SimulatedDrive]
    out_dir: Path
    frames: dict[str, pd.DataFrame]
    labels: pd.DataFrame


def derive_drive(drive: SimulatedDrive) -> pd.DataFrame:
    """Run the real chain message -> raw row -> ``derive_frames`` on one drive."""
    return derive_frames(raw_from_frames(drive.frames))


def labels_table(drives: list[SimulatedDrive]) -> pd.DataFrame:
    """Collect the fault labels of all drives into one table."""
    rows = [vars(label) for drive in drives for label in drive.labels]
    return pd.DataFrame(rows, columns=[*KEY, "kind", "expected_code"])


def missed_labels(frames: dict[str, pd.DataFrame], labels: pd.DataFrame) -> pd.DataFrame:
    """Return the labels with an expected code that the derived frames do not carry.

    Args:
        frames: Derived frames per ``drive_id``.
        labels: The fault labels.

    Returns:
        The labels whose frame is absent or carries a code other than ``expected_code``.
    """
    expected = labels[labels["expected_code"].notna()]
    table = pd.concat(frames.values(), ignore_index=True)[[*KEY, "error_code"]]
    found = expected.merge(table, on=KEY, how="left")
    return found[found["error_code"] != found["expected_code"]]


def blank_labeled_codes(frames: dict[str, pd.DataFrame], labels: pd.DataFrame) -> dict[str, Any]:
    """Return a copy of the derived frames with the codes of all labeled frames set to None."""
    out = {}
    for drive_id, table in frames.items():
        copy = table.copy()
        copy["error_code"] = copy["error_code"].astype(object)
        mine = labels[labels["drive_id"] == drive_id]["frame_idx"]
        copy.loc[copy["frame_idx"].isin(mine), "error_code"] = None
        out[drive_id] = copy
    return out


@pytest.fixture(scope="module")
def fleet(tmp_path_factory: pytest.TempPathFactory) -> Fleet:
    """Simulate the default fleet, write it to disk and derive every drive's frames."""
    drives = simulate_fleet(N_DRIVES, seed=FLEET_SEED, n_frames=N_FRAMES)
    out_dir = tmp_path_factory.mktemp("fleet")
    write_fleet_jsonl(drives, str(out_dir))
    frames = {drive.meta.drive_id: derive_drive(drive) for drive in drives}
    return Fleet(drives, out_dir, frames, labels_table(drives))


def check_scripted_drive(seed: int) -> None:
    """Assert one noisy, jittered, fault-free scripted drive gives the three scenario events."""
    sim = SimConfig(
        noise_sigma_cm=0.2,
        ts_jitter_ms=1.0,
        drop_burst_rate=0.0,
        duplicate_rate=0.0,
        sparse_rate=0.0,
        jump_rate=0.0,
        malformed_rate=0.0,
        late_arrival_rate=0.0,
    )
    drive = simulate_drive("s", "scripted", "car", N_FRAMES, seed, sim)
    frames = derive_drive(drive)
    events = detect_events(frames)
    summary = summarize_drive(frames, events).iloc[0]
    assert list(events["event_type"]) == ["HARD_BRAKE", "SHARP_TURN", "STOP"], f"seed {seed}"
    assert frames["error_code"].isna().all(), f"seed {seed}"
    assert summary["n_events"] == 3, f"seed {seed}"
    assert summary["distance_m"] == pytest.approx(SCRIPTED_DISTANCE_M, rel=0.02), f"seed {seed}"
    starts = events.set_index("event_type")["start_frame"]
    assert abs(starts["HARD_BRAKE"] - HARD_BRAKE_FRAME) <= TOLERANCE_FRAMES, f"seed {seed}"
    assert abs(starts["SHARP_TURN"] - SHARP_TURN_FRAME) <= TOLERANCE_FRAMES, f"seed {seed}"
    assert events.iloc[-1]["end_frame"] == frames["frame_idx"].iloc[-1], f"seed {seed}"


def test_scripted_drive_has_exactly_the_three_scenario_events() -> None:
    """Noise and jitter on, faults off: HARD_BRAKE, SHARP_TURN, STOP for each of 5 seeds."""
    for seed in range(5):
        check_scripted_drive(seed)


def test_fleet_validator_cli_passes_with_labels(
    fleet: Fleet, capsys: pytest.CaptureFixture[str]
) -> None:
    """The validator CLI exits 0 on the written fleet, with the label checks passing."""
    argv = ["--source", str(fleet.out_dir / "frames.jsonl")]
    argv += ["--labels", str(fleet.out_dir / "labels.json")]
    code = validate_main(argv)
    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert not [line for line in lines if line.startswith("FAIL")]
    assert any(line.startswith("PASS labeled_faults_detected:") for line in lines)
    assert any(line.startswith("PASS false_positive_rate:") for line in lines)


def test_every_labeled_fault_is_detected_with_its_code(fleet: Fleet) -> None:
    """Recall is 100%: each labeled fault frame carries exactly its expected code."""
    assert set(fleet.labels["kind"]) == FAULT_KINDS
    coded = fleet.labels["expected_code"].notna()
    assert coded.sum() > 0
    assert missed_labels(fleet.frames, fleet.labels).empty
    assert set(fleet.labels[coded]["expected_code"]) == {
        ErrorCode.DROPPED_FRAME.value,
        ErrorCode.POSITION_JUMP.value,
        ErrorCode.DUPLICATE_FRAME.value,
        ErrorCode.SPARSE_CLOUD.value,
    }


def test_false_positive_rate_stays_at_or_below_one_percent(fleet: Fleet) -> None:
    """Over all unlabeled frames at most ``max_false_positive_rate`` carry an error code."""
    table = pd.concat(fleet.frames.values(), ignore_index=True)
    marked = table.merge(fleet.labels[KEY].assign(labeled=True), on=KEY, how="left")
    unlabeled = marked[marked["labeled"].isna()]
    assert len(unlabeled) > N_DRIVES * N_FRAMES * 0.9
    assert unlabeled["error_code"].notna().mean() <= DEFAULT.max_false_positive_rate


def test_realism_report_passes_for_every_drive(fleet: Fleet) -> None:
    """Each drive's derived frames respect the physical limits of its profile and vehicle."""
    labels_json = json.loads((fleet.out_dir / "labels.json").read_text())
    assert len(labels_json["drives"]) == N_DRIVES
    for meta in labels_json["drives"]:
        report = validate_realism(
            fleet.frames[meta["drive_id"]], meta["profile"], meta["vehicle_kind"]
        )
        assert report.passed, f"{meta['drive_id']}\n{report.to_text()}"


def test_no_event_spans_a_flagged_frame_and_nothing_is_impossible(fleet: Fleet) -> None:
    """Events never contain a flagged frame, VL-003 never appears and durations are sane."""
    for frames in fleet.frames.values():
        assert not (frames["error_code"] == ErrorCode.IMPOSSIBLE_ACCEL.value).any()
        flagged = frames.loc[frames["error_code"].notna(), "frame_idx"]
        for event in detect_events(frames).itertuples():
            assert not flagged.between(event.start_frame, event.end_frame).any()
            duration = event.end_ts - event.start_ts
            assert math.isfinite(duration) and duration >= 0


def test_pipeline_is_deterministic() -> None:
    """The same seed gives identical derived frames and events through the whole chain."""
    runs = []
    for _ in range(2):
        drives = simulate_fleet(2, seed=FLEET_SEED, n_frames=500)
        frames = [derive_drive(drive) for drive in drives]
        runs.append((frames, [detect_events(table) for table in frames]))
    for first, second in zip(runs[0][0] + runs[0][1], runs[1][0] + runs[1][1], strict=True):
        assert_frame_equal(first, second)


def test_sanity_detects_a_broken_pipeline(fleet: Fleet) -> None:
    """Mutation guard: blanking the codes of labeled frames makes the label check report misses."""
    expected = fleet.labels["expected_code"].notna().sum()
    broken = blank_labeled_codes(fleet.frames, fleet.labels)
    missed = missed_labels(broken, fleet.labels)
    assert len(missed) == expected > 0
