"""Tests for driving profiles: the clean ground-truth track."""

import math

import numpy as np
import numpy.typing as npt

from sim_helpers import DT
from vl.config import DEFAULT
from vl.mock.profiles import PROFILE_LIMITS, Track, generate_track

PROFILES = ["highway", "city", "curvy", "parking"]
TOLERANCE = 1.01  # 1% slack on each limit for float and chord effects


def speeds(track: Track) -> npt.NDArray[np.float64]:
    """Finite-difference speed from the positions, in m/s."""
    return np.hypot(np.diff(track.x), np.diff(track.y)) / DT


def yaw_rates(track: Track) -> npt.NDArray[np.float64]:
    """Finite-difference yaw rate from the headings, wrapped to (-pi, pi], in rad/s."""
    step = np.diff(track.heading)
    return ((step + math.pi) % (2 * math.pi) - math.pi) / DT


def longest_run(mask: npt.NDArray[np.bool_]) -> int:
    """Length of the longest run of consecutive True values."""
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def test_profiles_within_limits() -> None:
    for profile in PROFILES:
        limits = PROFILE_LIMITS[profile]
        for seed in range(3):
            case = (profile, seed)
            track = generate_track(profile, 4000, seed)
            speed = speeds(track)
            assert speed.max() <= limits.max_speed_ms * TOLERANCE, case
            assert np.abs(np.diff(speed) / DT).max() <= limits.max_abs_accel_ms2 * TOLERANCE, case
            assert np.abs(yaw_rates(track)).max() <= limits.max_abs_yaw_rate_rad_s * TOLERANCE, case


def test_city_has_a_long_stop() -> None:
    track = generate_track("city", 16000, 0)
    stopped = speeds(track) < DEFAULT.stop_speed_ms
    assert longest_run(stopped) * DEFAULT.frame_period_s >= 5.0


def test_highway_mean_speed_above_20() -> None:
    for seed in range(3):
        track = generate_track("highway", 8000, seed)
        assert speeds(track).mean() > 20.0, seed


def test_tracks_have_n_frames_and_are_deterministic_by_seed() -> None:
    for profile in ["scripted", *PROFILES]:
        track = generate_track(profile, 500, 0)
        assert {len(track.x), len(track.y), len(track.speed), len(track.heading)} == {500}, profile
    first, again = generate_track("curvy", 500, 5), generate_track("curvy", 500, 5)
    other = generate_track("curvy", 500, 6)
    assert np.array_equal(first.x, again.x) and np.array_equal(first.y, again.y)
    assert not np.array_equal(first.x, other.x)


def test_profiles_are_alive() -> None:
    # Liveness guard: a stationary or straight track must not satisfy the limits trivially.
    for profile, min_mean_speed, min_peak_yaw in [("curvy", 5.0, 0.1), ("parking", 0.5, 0.2)]:
        for seed in range(3):
            track = generate_track(profile, 4000, seed)
            assert speeds(track).mean() > min_mean_speed, (profile, seed)
            assert np.abs(yaw_rates(track)).max() > min_peak_yaw, (profile, seed)


def test_track_speed_matches_position_differences() -> None:
    # Alignment-agnostic: the chord speed between frames k and k+1 must lie between the two
    # recorded speeds, give or take 1% + 0.05 m/s.
    for profile in PROFILES:
        track = generate_track(profile, 4000, 0)
        chord = speeds(track)
        low = np.minimum(track.speed[:-1], track.speed[1:])
        high = np.maximum(track.speed[:-1], track.speed[1:])
        slack = 0.01 * high + 0.05
        assert np.all(chord >= low - slack) and np.all(chord <= high + slack), profile


def test_ordinary_acceleration_stays_below_the_hard_accel_threshold() -> None:
    for profile in ["city", "curvy"]:
        for seed in range(10):
            accel = np.diff(speeds(generate_track(profile, 4000, seed))) / DT
            assert accel.max() <= 0.8 * DEFAULT.hard_accel_ms2 * 1.05, (profile, seed)


def test_positions_use_the_trapezoid_rule() -> None:
    # Scripted frames 0..100 accelerate on a straight line: the chord between two frames is
    # the mean of the two recorded speeds, not the speed of the first one.
    track = generate_track("scripted", 1000, 0)
    chord = speeds(track)[:100]
    mean_speed = (track.speed[:100] + track.speed[1:101]) / 2.0
    assert np.allclose(chord, mean_speed, atol=1e-9)
