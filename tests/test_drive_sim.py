"""Tests for the drive simulator: config, geometry, noise, jitter, faults and fleets."""

from dataclasses import fields, replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from sim_helpers import DT, centroids_cm, frame_indices
from vl.mock.drive_sim import DriveMeta, SimConfig, SimulatedDrive, simulate_drive, simulate_fleet
from vl.mock.faults import Fault, FaultLabel
from vl.mock.profiles import generate_track
from vl.mock.vehicles import VEHICLES

CLEAN = SimConfig.clean()
FLEET_PROFILES = {"highway", "city", "curvy", "parking"}
META_FIELDS = {
    "drive_id",
    "vehicle_id",
    "vehicle_kind",
    "profile",
    "drive_start_utc",
    "n_frames",
    "seed",
}


def test_sim_config_clean_is_all_zero() -> None:
    assert all(getattr(CLEAN, field.name) == 0.0 for field in fields(SimConfig))


def test_scripted_contract_geometry() -> None:
    drive = simulate_drive("d", "scripted", "car", 1000, 0, CLEAN)
    assert len(drive.frames) == 1000
    assert drive.frames[16]["ts"] == 1.0
    centers = centroids_cm(drive.frames)
    speed = np.hypot(*(centers[248, :2] - centers[232, :2])) / 100.0 / (16 * DT)
    assert speed == pytest.approx(15.0, abs=0.05)


def test_deterministic_by_seed() -> None:
    first = simulate_drive("d", "city", "suv", 300, 3)
    assert first == simulate_drive("d", "city", "suv", 300, 3)
    assert first.frames != simulate_drive("d", "city", "suv", 300, 4).frames


def test_noise_statistics() -> None:
    sim = replace(CLEAN, noise_sigma_cm=0.2, ts_jitter_ms=0.0)  # only noise is on
    drive = simulate_drive("d", "city", "car", 1000, 7, sim)
    track = generate_track("city", 1000, 7)
    truth = 100.0 * np.stack([track.x, track.y], axis=1)
    error = centroids_cm(drive.frames)[:, :2] - truth
    assert len(drive.frames) == 1000
    assert np.all((error.std(axis=0) >= 0.15) & (error.std(axis=0) <= 0.25))


def test_positions_follow_jittered_ts() -> None:
    # Note for the implementer: positions are sampled at the jittered ts. Sampling at nominal
    # times would leave errors of centimeters at highway speed, so this test discriminates.
    # The first and last frames are skipped: ts may fall outside the track (extrapolation).
    sim = replace(CLEAN, noise_sigma_cm=0.0, ts_jitter_ms=1.0)
    drive = simulate_drive("d", "highway", "car", 1000, 0, sim)
    track = generate_track("highway", 1000, 0)
    ts = np.array([frame["ts"] for frame in drive.frames])[1:-1]
    nominal = np.arange(1000) * DT
    truth = 100.0 * np.stack(
        [np.interp(ts, nominal, track.x), np.interp(ts, nominal, track.y)], axis=1
    )
    error = centroids_cm(drive.frames)[1:-1, :2] - truth
    assert np.abs(error).max() <= 0.5


def test_ts_jitter() -> None:
    drive = simulate_drive("d", "city", "car", 1000, 7, replace(CLEAN, ts_jitter_ms=1.0))
    ts = np.array([frame["ts"] for frame in drive.frames])
    jitter_ms = (ts - np.arange(1000) * DT) * 1000.0
    assert 0.7 <= jitter_ms.std() <= 1.3
    assert np.abs(jitter_ms).max() <= 5.0
    assert np.all(np.diff(ts) > 0)


def test_late_arrival_changes_order_not_content() -> None:
    # Note for the implementer: late-arrival draws must come from a stream independent of the
    # noise/dither draws (this test compares against a rate-0 drive of the same seed).
    late = simulate_drive("d", "city", "car", 300, 4, replace(CLEAN, late_arrival_rate=0.05))
    clean = simulate_drive("d", "city", "car", 300, 4, CLEAN)
    order = frame_indices(late.frames)
    assert sorted(order) == list(range(300)) and order != sorted(order)
    assert sorted(late.frames, key=lambda f: f["frame_idx"]) == clean.frames
    assert late.labels == []


def test_simulate_drive_applies_explicit_faults() -> None:
    faults = (Fault("drop", 10, 2), Fault("duplicate", 50))
    drive = simulate_drive("d", "scripted", "car", 100, 0, CLEAN, faults)
    assert frame_indices(drive.frames) == [*range(10), *range(12, 51), *range(50, 100)]
    assert sorted(drive.labels, key=lambda label: label.frame_idx) == [
        FaultLabel("d", 12, "drop", "VL-001"),
        FaultLabel("d", 50, "duplicate", "VL-004"),
    ]
    past_end = simulate_drive("d", "scripted", "car", 100, 0, CLEAN, (Fault("jump", 500),))
    assert past_end.labels == [] and len(past_end.frames) == 100


def is_periodic(values: list[str], period: int) -> bool:
    return all(values[i] == values[i + period] for i in range(len(values) - period))


def check_cycles(fleet: list[SimulatedDrive]) -> None:
    kinds = [drive.meta.vehicle_kind for drive in fleet]
    profiles = [drive.meta.profile for drive in fleet]
    assert set(kinds) == set(VEHICLES) and is_periodic(kinds, len(set(kinds)))
    assert set(profiles) == FLEET_PROFILES and is_periodic(profiles, len(set(profiles)))


def check_meta(drive: SimulatedDrive, n_frames: int) -> None:
    meta = drive.meta
    assert isinstance(meta.vehicle_id, str) and meta.vehicle_id
    assert isinstance(meta.seed, int) and meta.n_frames == n_frames
    for frame in drive.frames:
        assert frame["drive_id"] == meta.drive_id
        assert frame["vehicle_id"] == meta.vehicle_id
        assert frame["vehicle_kind"] == meta.vehicle_kind
        assert frame["drive_start_utc"] == meta.drive_start_utc


def test_fleet_metadata_and_determinism() -> None:
    fleet = simulate_fleet(15, seed=3, n_frames=50)
    ids = [drive.meta.drive_id for drive in fleet]
    assert ids == [f"drive-{i:03d}" for i in range(1, 16)]
    check_cycles(fleet)
    assert {field.name for field in fields(DriveMeta)} == META_FIELDS
    for drive in fleet:
        check_meta(drive, 50)
    starts = [datetime.fromisoformat(drive.meta.drive_start_utc) for drive in fleet]
    assert max(starts) - min(starts) < timedelta(hours=24) and len(set(starts)) > 1
    assert simulate_fleet(15, seed=3, n_frames=50) == fleet


def gap_labels(drive: SimulatedDrive) -> list[FaultLabel]:
    return [label for label in drive.labels if label.kind == "gap_after_malformed"]


def test_malformed_adds_a_gap_label_on_the_next_frame() -> None:
    drive = simulate_drive("d", "scripted", "car", 100, 0, CLEAN, (Fault("malformed", 20),))
    assert sorted(drive.labels, key=lambda label: label.frame_idx) == [
        FaultLabel("d", 20, "malformed", None),
        FaultLabel("d", 21, "gap_after_malformed", "VL-001"),
    ]
    last = simulate_drive("d", "scripted", "car", 100, 0, CLEAN, (Fault("malformed", 99),))
    assert last.labels == [FaultLabel("d", 99, "malformed", None)]  # no next frame, no gap label


def test_random_malformed_faults_all_get_gap_labels() -> None:
    sim = replace(CLEAN, malformed_rate=0.05)
    drive = simulate_drive("d", "city", "car", 400, 2, sim)
    malformed = [label.frame_idx for label in drive.labels if label.kind == "malformed"]
    gaps = [label.frame_idx for label in gap_labels(drive)]
    assert malformed and gaps == [idx + 1 for idx in malformed if idx < 399]


def test_colliding_explicit_faults_raise_from_simulate_drive() -> None:
    faults = (Fault("drop", 10, 3), Fault("duplicate", 13))
    with pytest.raises(ValueError, match="duplicate"):
        simulate_drive("d", "scripted", "car", 100, 0, CLEAN, faults)
