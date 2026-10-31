"""Driving profiles: the ground-truth vehicle track (position, speed, heading) per frame.

Role in the pipeline: the simulator samples a clean `Track` here, then adds sensor noise and
faults on top of it.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from vl.config import DEFAULT, Config


@dataclass(frozen=True)
class ProfileLimits:
    """Physical limits a profile's clean track must never exceed.

    Attributes:
        max_speed_ms: Largest speed in m/s.
        max_abs_accel_ms2: Largest absolute acceleration in m/s^2.
        max_abs_yaw_rate_rad_s: Largest absolute yaw rate in rad/s.
    """

    max_speed_ms: float
    max_abs_accel_ms2: float
    max_abs_yaw_rate_rad_s: float


PROFILE_LIMITS: dict[str, ProfileLimits] = {
    "scripted": ProfileLimits(20.0, 5.5, 0.55),
    "highway": ProfileLimits(38.0, 4.0, 0.08),
    "city": ProfileLimits(17.0, 8.0, 0.6),
    "curvy": ProfileLimits(25.0, 5.0, 0.3),
    "parking": ProfileLimits(4.0, 2.0, 0.8),
}


@dataclass
class Track:
    """Per-frame ground truth of one drive.

    Attributes:
        x: Position along x in meters, one value per frame.
        y: Position along y in meters, one value per frame.
        speed: Speed in m/s, one value per frame.
        heading: Heading in radians, one value per frame.
    """

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    speed: npt.NDArray[np.float64]
    heading: npt.NDArray[np.float64]


Floats = npt.NDArray[np.float64]

_SAFETY = 0.9  # random maneuvers use at most this fraction of a profile's limits
_FUZZ = 1e-9  # guards ceil() against float noise such as 128.00000000000003
_NO_YAW = 0.0


class _Segment(NamedTuple):
    """A stretch of the drive: per-frame speed (m/s) and yaw rate (rad/s)."""

    speed: Floats
    yaw: Floats


@dataclass(frozen=True)
class _Shape:
    """Driving style of one random profile.

    Attributes:
        low_speed: Below this speed the vehicle always accelerates next, in m/s.
        high_speed: Cruising and acceleration targets never exceed this, in m/s.
        cruise_s: Range of cruise durations in seconds.
        turn_speed: Range of the speed taken into an intersection turn, in m/s.
        weights: Relative weight of each maneuver name.
    """

    low_speed: float
    high_speed: float
    cruise_s: tuple[float, float]
    turn_speed: tuple[float, float]
    weights: dict[str, float]


@dataclass
class _State:
    """What a maneuver generator needs to know.

    Attributes:
        rng: The track's random generator.
        speed: Current speed in m/s.
        limits: The profile's physical limits.
        shape: The profile's driving style.
        cfg: Pipeline thresholds (frame period, event thresholds).
    """

    rng: np.random.Generator
    speed: float
    limits: ProfileLimits
    shape: _Shape
    cfg: Config


_SHAPES: dict[str, _Shape] = {
    "highway": _Shape(
        24.0,
        34.0,
        (6.0, 20.0),
        (0.0, 0.0),
        {
            "cruise": 5.0,
            "accelerate": 1.5,
            "brake": 1.5,
            "hard_brake": 0.3,
            "lane_change": 1.5,
            "gentle_curve": 0.5,
        },
    ),
    "city": _Shape(
        8.0,
        15.0,
        (2.0, 8.0),
        (3.0, 6.0),
        {
            "cruise": 2.0,
            "accelerate": 2.0,
            "brake": 1.5,
            "hard_brake": 0.5,
            "stop_at_light": 1.5,
            "intersection_turn": 1.5,
            "lane_change": 0.3,
            "gentle_curve": 0.3,
        },
    ),
    "curvy": _Shape(
        10.0,
        22.0,
        (2.0, 6.0),
        (5.0, 10.0),
        {
            "cruise": 1.5,
            "accelerate": 1.5,
            "brake": 1.5,
            "hard_brake": 0.3,
            "intersection_turn": 1.0,
            "lane_change": 0.5,
            "gentle_curve": 4.0,
        },
    ),
    "parking": _Shape(
        1.0,
        3.5,
        (1.0, 4.0),
        (1.0, 2.0),
        {
            "cruise": 1.5,
            "accelerate": 2.0,
            "brake": 1.5,
            "stop_at_light": 0.5,
            "intersection_turn": 2.5,
            "gentle_curve": 1.5,
        },
    ),
}


def _frames(seconds: float, dt: float) -> int:
    """Convert a duration to a frame count of at least one.

    Args:
        seconds: Duration in seconds.
        dt: Frame period in seconds.

    Returns:
        The number of frames, rounded to nearest, never below 1.
    """
    return max(1, round(seconds / dt))


def _ramp(v0: float, v1: float, accel: float, dt: float) -> Floats:
    """Speeds of a constant-acceleration change from ``v0`` to ``v1``.

    Args:
        v0: Starting speed in m/s (not included in the result).
        v1: Final speed in m/s (the last value of the result).
        accel: Magnitude of the acceleration in m/s^2.
        dt: Frame period in seconds.

    Returns:
        One speed per frame, empty when ``v0 == v1``.
    """
    steps = math.ceil(abs(v1 - v0) / (accel * dt) - _FUZZ)
    values = v0 + math.copysign(accel * dt, v1 - v0) * np.arange(1, steps + 1)
    return np.clip(values, min(v0, v1), max(v0, v1))


def _straight(speed: Floats) -> _Segment:
    """Wrap per-frame speeds into a segment without turning.

    Args:
        speed: Per-frame speeds in m/s.

    Returns:
        The segment with a zero yaw rate throughout.
    """
    return _Segment(speed, np.full(len(speed), _NO_YAW))


def _hold(speed: float, seconds: float, dt: float) -> _Segment:
    """A straight stretch at constant speed.

    Args:
        speed: The constant speed in m/s.
        seconds: Duration in seconds.
        dt: Frame period in seconds.

    Returns:
        The segment.
    """
    return _straight(np.full(_frames(seconds, dt), speed))


def _turn(speed: float, yaw_rate: float, seconds: float, dt: float) -> _Segment:
    """A constant-speed, constant-yaw-rate stretch.

    Args:
        speed: The constant speed in m/s.
        yaw_rate: Signed yaw rate in rad/s.
        seconds: Duration in seconds.
        dt: Frame period in seconds.

    Returns:
        The segment.
    """
    n = _frames(seconds, dt)
    return _Segment(np.full(n, speed), np.full(n, yaw_rate))


def _join(first: _Segment, second: _Segment) -> _Segment:
    """Concatenate two segments.

    Args:
        first: The earlier segment.
        second: The later segment.

    Returns:
        The combined segment.
    """
    return _Segment(
        np.concatenate([first.speed, second.speed]), np.concatenate([first.yaw, second.yaw])
    )


def _sign(st: _State) -> float:
    """Draw a random turning direction.

    Args:
        st: Maneuver state.

    Returns:
        ``-1.0`` or ``1.0``.
    """
    return float(st.rng.choice([-1.0, 1.0]))


def _accel_cap(st: _State) -> float:
    """Largest acceleration magnitude a random maneuver may use.

    Args:
        st: Maneuver state.

    Returns:
        A fraction of the profile's limit, in m/s^2.
    """
    return _SAFETY * st.limits.max_abs_accel_ms2


def _yaw_cap(st: _State) -> float:
    """Largest yaw rate magnitude a random maneuver may use.

    Args:
        st: Maneuver state.

    Returns:
        A fraction of the profile's limit, in rad/s.
    """
    return _SAFETY * st.limits.max_abs_yaw_rate_rad_s


def _cruise(st: _State) -> _Segment:
    """Maneuver: hold the current speed.

    Args:
        st: Maneuver state.

    Returns:
        The segment.
    """
    return _hold(st.speed, st.rng.uniform(*st.shape.cruise_s), st.cfg.frame_period_s)


def _accelerate(st: _State) -> _Segment:
    """Maneuver: speed up to a new cruising speed.

    Args:
        st: Maneuver state.

    Returns:
        The segment (empty when the vehicle is already at the top speed).
    """
    shape = st.shape
    target = st.rng.uniform(
        min(max(st.speed + 1.0, shape.low_speed), shape.high_speed), shape.high_speed
    )
    comfort = 0.8 * st.cfg.hard_accel_ms2
    accel = st.rng.uniform(0.3, 1.0) * min(_accel_cap(st), comfort)
    return _straight(_ramp(st.speed, target, accel, st.cfg.frame_period_s))


def _brake(st: _State) -> _Segment:
    """Maneuver: ordinary braking, below the hard-brake threshold.

    Args:
        st: Maneuver state.

    Returns:
        The segment.
    """
    target = st.speed * st.rng.uniform(0.5, 0.9)
    comfort = 0.8 * abs(st.cfg.hard_brake_ms2)
    accel = st.rng.uniform(0.3, 1.0) * min(_accel_cap(st), comfort)
    return _straight(_ramp(st.speed, target, accel, st.cfg.frame_period_s))


def _hard_brake(st: _State) -> _Segment:
    """Maneuver: the strongest braking the profile allows.

    Args:
        st: Maneuver state.

    Returns:
        The segment.
    """
    target = st.speed * st.rng.uniform(0.0, 0.4)
    accel = st.rng.uniform(0.7, 1.0) * _accel_cap(st)
    return _straight(_ramp(st.speed, target, accel, st.cfg.frame_period_s))


def _stop_at_light(st: _State) -> _Segment:
    """Maneuver: brake to a standstill and wait for 5 to 30 seconds.

    Args:
        st: Maneuver state.

    Returns:
        The segment.
    """
    dt = st.cfg.frame_period_s
    accel = min(st.rng.uniform(1.5, 3.0), _accel_cap(st))
    stop = _straight(_ramp(st.speed, 0.0, accel, dt))
    return _join(stop, _hold(0.0, st.rng.uniform(5.0, 30.0), dt))


def _intersection_turn(st: _State) -> _Segment:
    """Maneuver: slow down and turn through about 90 degrees.

    Args:
        st: Maneuver state.

    Returns:
        The segment.
    """
    dt = st.cfg.frame_period_s
    speed = min(st.speed, st.rng.uniform(*st.shape.turn_speed))
    slow = _straight(_ramp(st.speed, speed, min(2.0, _accel_cap(st)), dt))
    rate = st.rng.uniform(0.5, 1.0) * _yaw_cap(st)
    return _join(slow, _turn(speed, _sign(st) * rate, (math.pi / 2.0) / rate, dt))


def _bump(st: _State, amplitude: float, seconds: float, cycles: float) -> _Segment:
    """A straight-speed stretch whose yaw rate follows a sine wave.

    Args:
        st: Maneuver state.
        amplitude: Peak yaw rate magnitude in rad/s.
        seconds: Duration in seconds.
        cycles: Sine periods covered (0.5 is one bump, 1.0 an S-curve).

    Returns:
        The segment at the current speed.
    """
    n = _frames(seconds, st.cfg.frame_period_s)
    phase = 2.0 * math.pi * cycles * (np.arange(n) + 0.5) / n
    return _Segment(np.full(n, st.speed), _sign(st) * amplitude * np.sin(phase))


def _lane_change(st: _State) -> _Segment:
    """Maneuver: an S-shaped sideways shift with no net heading change.

    Args:
        st: Maneuver state.

    Returns:
        The segment.
    """
    amplitude = st.rng.uniform(0.3, 0.7) * _yaw_cap(st)
    return _bump(st, amplitude, st.rng.uniform(3.0, 5.0), 1.0)


def _gentle_curve(st: _State) -> _Segment:
    """Maneuver: a smooth bend in one direction.

    Args:
        st: Maneuver state.

    Returns:
        The segment.
    """
    amplitude = st.rng.uniform(0.3, 0.9) * _yaw_cap(st)
    return _bump(st, amplitude, st.rng.uniform(4.0, 12.0), 0.5)


_MANEUVERS: dict[str, Callable[[_State], _Segment]] = {
    "cruise": _cruise,
    "accelerate": _accelerate,
    "brake": _brake,
    "hard_brake": _hard_brake,
    "stop_at_light": _stop_at_light,
    "intersection_turn": _intersection_turn,
    "lane_change": _lane_change,
    "gentle_curve": _gentle_curve,
}


def _pick(st: _State) -> str:
    """Choose the next maneuver: speed up when too slow, else by profile weights.

    Args:
        st: Maneuver state.

    Returns:
        A key of ``_MANEUVERS``.
    """
    shape = st.shape
    if st.speed < shape.low_speed:
        return "accelerate"
    names = list(shape.weights)
    weights = np.array([shape.weights[name] for name in names])
    return names[int(st.rng.choice(len(names), p=weights / weights.sum()))]


def _random_segments(profile: str, n_frames: int, seed: int, cfg: Config) -> list[_Segment]:
    """Chain seeded maneuvers until at least ``n_frames`` frames are covered.

    Args:
        profile: A random profile (a key of ``_SHAPES``).
        n_frames: Frames to cover.
        seed: Random seed.
        cfg: Pipeline thresholds.

    Returns:
        The maneuver segments in driving order.
    """
    st = _State(np.random.default_rng(seed), 0.0, PROFILE_LIMITS[profile], _SHAPES[profile], cfg)
    segments: list[_Segment] = []
    covered = 0
    while covered < n_frames:
        segment = _MANEUVERS[_pick(st)](st)
        segments.append(segment)
        covered += len(segment.speed)
        st.speed = float(segment.speed[-1]) if len(segment.speed) else st.speed
    return segments


def _scripted_segments(dt: float) -> list[_Segment]:
    """The deterministic scenario: one hard brake, one sharp turn, one stop.

    Accelerate 0 to 15 m/s over 8 s, cruise 12 s, brake at -5 m/s^2 for 2 s, cruise 5 s, turn
    at 0.5 rad/s for 3 s, accelerate to 15 m/s over 4 s, cruise 10 s, decelerate at -2 m/s^2
    to a stop (the caller pads the rest of the drive with standstill).

    Args:
        dt: Frame period in seconds.

    Returns:
        The scenario segments in driving order.
    """
    return [
        _straight(_ramp(0.0, 15.0, 1.875, dt)),
        _hold(15.0, 12.0, dt),
        _straight(_ramp(15.0, 5.0, 5.0, dt)),
        _hold(5.0, 5.0, dt),
        _turn(5.0, 0.5, 3.0, dt),
        _straight(_ramp(5.0, 15.0, 2.5, dt)),
        _hold(15.0, 10.0, dt),
        _straight(_ramp(15.0, 0.0, 2.0, dt)),
    ]


def _assemble(segments: list[_Segment], n_frames: int, dt: float) -> Track:
    """Integrate segments into a track of exactly ``n_frames`` frames.

    ``speed[k]`` and ``heading[k]`` are the velocity at frame ``k``; positions are the
    trapezoid integral of the velocity, so the chord between consecutive positions is the mean
    of the two recorded speeds (up to the small heading change). A drive shorter than
    ``n_frames`` is padded with standstill.

    Args:
        segments: Segments in driving order.
        n_frames: Frames in the track.
        dt: Frame period in seconds.

    Returns:
        The integrated `Track`.
    """
    speed = np.concatenate([s.speed for s in segments] + [np.zeros(n_frames)])[:n_frames]
    yaw = np.concatenate([s.yaw for s in segments] + [np.zeros(n_frames)])[:n_frames]
    heading = np.concatenate([[0.0], np.cumsum(yaw * dt)[:-1]])
    x = _trapezoid(speed * np.cos(heading), dt)
    y = _trapezoid(speed * np.sin(heading), dt)
    return Track(x, y, speed, heading)


def _trapezoid(velocity: Floats, dt: float) -> Floats:
    """Integrate a per-frame velocity with the trapezoid rule, starting at zero.

    Args:
        velocity: Velocity component at every frame, in m/s.
        dt: Frame period in seconds.

    Returns:
        The position at every frame, in meters (first value 0).
    """
    steps = 0.5 * dt * (velocity[:-1] + velocity[1:])
    return np.concatenate([[0.0], np.cumsum(steps)])


def generate_track(profile: str, n_frames: int, seed: int, cfg: Config = DEFAULT) -> Track:
    """Generate the clean ground-truth track of one drive.

    ``scripted`` is the deterministic scenario (``seed`` is ignored); the other profiles are
    seeded sequences of maneuvers that never exceed ``PROFILE_LIMITS[profile]``.

    Args:
        profile: A key of `PROFILE_LIMITS`.
        n_frames: Number of frames in the track.
        seed: Random seed; the same seed gives an identical track.
        cfg: Pipeline thresholds; ``cfg.frame_period_s`` is the time step.

    Returns:
        A `Track` whose arrays all have length ``n_frames``.

    Raises:
        ValueError: If ``profile`` is not a key of `PROFILE_LIMITS`.
    """
    dt = cfg.frame_period_s
    if profile == "scripted":
        return _assemble(_scripted_segments(dt), n_frames, dt)
    if profile not in _SHAPES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(PROFILE_LIMITS)}")
    return _assemble(_random_segments(profile, n_frames, seed, cfg), n_frames, dt)
