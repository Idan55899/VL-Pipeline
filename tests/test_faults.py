"""Tests for explicit fault injection, its labels, and random fault rates."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from sim_helpers import DT, Frame, frame_indices, make_frames
from vl.mock.drive_sim import SimConfig
from vl.mock.faults import Fault, FaultLabel, apply_faults, random_faults

VALID_KINDS = {"drop", "jump", "duplicate", "sparse", "malformed", "ts_backwards"}
RATE_FIELDS = {
    "drop": "drop_burst_rate",
    "duplicate": "duplicate_rate",
    "sparse": "sparse_rate",
    "jump": "jump_rate",
    "malformed": "malformed_rate",
}


def apply_one(fault: Fault) -> tuple[list[Frame], list[FaultLabel]]:
    return apply_faults(make_frames(), (fault,), "d")


def test_drop_burst_label_on_first_frame_after_gap() -> None:
    out, labels = apply_one(Fault("drop", 5, 3))
    assert frame_indices(out) == [*range(5), *range(8, 20)]
    assert labels == [FaultLabel("d", 8, "drop", "VL-001")]


def test_jump_shifts_this_and_all_later_frames_by_5m() -> None:
    reference = make_frames()
    out, labels = apply_one(Fault("jump", 10))
    assert out[:10] == reference[:10]
    for got, want in zip(out[10:], reference[10:], strict=True):
        assert got["points"] == [[x + 500, y, z] for x, y, z in want["points"]]
        assert got["ts"] == want["ts"] and got["frame_idx"] == want["frame_idx"]
    assert labels == [FaultLabel("d", 10, "jump", "VL-002")]


def test_duplicate_emits_frame_twice() -> None:
    reference = make_frames()
    out, labels = apply_one(Fault("duplicate", 7))
    assert frame_indices(out) == [*range(8), 7, *range(8, 20)]
    assert out[7] == out[8] == reference[7]
    assert out[7]["points"] is not out[8]["points"]  # the copy owns its points list
    assert labels == [FaultLabel("d", 7, "duplicate", "VL-004")]


def test_sparse_keeps_only_ten_points() -> None:
    reference = make_frames()
    out, labels = apply_one(Fault("sparse", 4))
    kept = {tuple(point) for point in out[4]["points"]}
    assert len(out[4]["points"]) == len(kept) == 10
    assert kept <= {tuple(point) for point in reference[4]["points"]}
    assert out[:4] + out[5:] == reference[:4] + reference[5:]
    assert labels == [FaultLabel("d", 4, "sparse", "VL-006")]


def test_malformed_replaces_points_with_garbage_and_has_no_code() -> None:
    reference = make_frames()
    out, labels = apply_one(Fault("malformed", 6))
    assert len(out) == 20 and out[6]["points"] == "garbage"
    assert out[:6] + out[7:] == reference[:6] + reference[7:]
    assert labels == [FaultLabel("d", 6, "malformed", None)]


def test_ts_backwards_is_previous_ts_minus_one_ms() -> None:
    reference = make_frames()
    out, labels = apply_one(Fault("ts_backwards", 9))
    assert out[9]["ts"] == pytest.approx(8 * DT - 0.001, abs=1e-12)
    assert out[:9] + out[10:] == reference[:9] + reference[10:]
    assert labels == [FaultLabel("d", 9, "ts_backwards", "VL-004")]


def pooled_counts(n_frames: int, n_rngs: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for seed in range(n_rngs):
        faults = random_faults(n_frames, SimConfig(), np.random.default_rng(seed))
        counts.update(fault.kind for fault in faults)
    return counts


def test_random_fault_rates() -> None:
    # Note for the implementer: the pooled bands are fixed by the plan. If the jump count
    # fails by chance (about 2%), re-seed or consume the rng differently; never loosen the band.
    n_frames, n_rngs = 20000, 5
    counts = pooled_counts(n_frames, n_rngs)
    for kind, rate_field in RATE_FIELDS.items():
        expected = getattr(SimConfig(), rate_field) * n_frames * n_rngs
        assert 0.5 * expected <= counts[kind] <= 1.5 * expected, kind


def test_random_faults_are_well_formed() -> None:
    faults = random_faults(20000, SimConfig(), np.random.default_rng(0))
    assert faults and {fault.kind for fault in faults} <= VALID_KINDS
    assert all(0 <= fault.frame_idx < 20000 for fault in faults)
    assert all(1 <= f.length <= 5 for f in faults if f.kind == "drop")
    assert all(f.length == 1 for f in faults if f.kind != "drop")


COLLISIONS = [
    (Fault("sparse", 4), Fault("duplicate", 4)),
    (Fault("drop", 5, 3), Fault("jump", 6)),
    (Fault("drop", 5, 3), Fault("drop", 7, 2)),
    (Fault("drop", 5, 3), Fault("sparse", 8)),
    (Fault("malformed", 6), Fault("duplicate", 7)),
    (Fault("malformed", 6), Fault("drop", 7)),
]


def test_colliding_faults_raise_value_error_naming_both() -> None:
    for first, second in COLLISIONS:
        with pytest.raises(ValueError) as caught:
            apply_faults(make_frames(), (second, first), "d")
        assert repr(first) in str(caught.value), (first, second)
        assert repr(second) in str(caught.value), (first, second)


def test_frame_zero_faults() -> None:
    for kind in ["drop", "jump", "ts_backwards"]:  # undetectable at frame 0
        with pytest.raises(ValueError, match=kind):
            apply_faults(make_frames(), (Fault(kind, 0),), "d")
    for kind in ["duplicate", "sparse", "malformed"]:  # allowed at frame 0
        _, labels = apply_faults(make_frames(), (Fault(kind, 0),), "d")
        assert [label.kind for label in labels] == [kind], kind


def test_fault_past_the_end_is_skipped_silently() -> None:
    out, labels = apply_one(Fault("jump", 99))
    assert out == make_frames() and labels == []


def test_jump_before_a_malformed_frame_does_not_crash() -> None:
    out, labels = apply_faults(make_frames(), (Fault("jump", 3), Fault("malformed", 6)), "d")
    assert out[6]["points"] == "garbage"
    assert [label.kind for label in labels] == ["jump", "malformed"]


def test_random_faults_never_collide_and_skip_the_frame_after_malformed() -> None:
    busy = replace(
        SimConfig(), drop_burst_rate=0.03, duplicate_rate=0.03, sparse_rate=0.03,
        jump_rate=0.03, malformed_rate=0.05,
    )  # fmt: skip
    for seed in range(5):
        faults = random_faults(1500, busy, np.random.default_rng(seed))
        apply_faults(make_frames(1500), faults, "d")  # raises on any collision
        starts = {f.frame_idx for f in faults}
        assert not any(f.frame_idx + 1 in starts for f in faults if f.kind == "malformed")
