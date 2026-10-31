"""Realistic drive simulator: tracks, noisy point clouds, faults, and fleets of drives.

Role in the pipeline: the mock data source. It turns a ground-truth track into frame messages
(``drive_id, vehicle_id, vehicle_kind, drive_start_utc, frame_idx, ts, points``) with sensor
noise, timestamp jitter, injected faults and fault labels.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from vl.config import DEFAULT, Config
from vl.mock.faults import Fault, FaultLabel, apply_faults, malformed_gap_labels, random_faults
from vl.mock.profiles import Floats, Track, generate_track
from vl.mock.vehicles import VEHICLES, cloud_lattice

_JITTER_CLIP_MS = 5.0  # keeps timestamps strictly increasing (the frame period is 62.5 ms)
_LATE_REACH = 2  # a late frame swaps with one up to this many positions ahead (window of 3)
_DAY_S = 24 * 3600
_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
_FLEET_PROFILES = ("highway", "city", "curvy", "parking")
_FLEET_VEHICLES = ("car", "suv", "truck")


@dataclass(frozen=True)
class SimConfig:
    """Sensor noise and fault rates of the simulator.

    Attributes:
        noise_sigma_cm: Gaussian noise sigma on the vehicle position per axis, in cm,
            realized by per-point stochastic rounding (dithering) to the cm grid.
        ts_jitter_ms: Gaussian timestamp jitter sigma in ms, clipped to +-5 ms.
        drop_burst_rate: Per-frame chance of a burst of 1 to 5 dropped frames.
        duplicate_rate: Per-frame chance of a duplicated frame.
        sparse_rate: Per-frame chance of a sparse cloud.
        jump_rate: Per-frame chance of a position jump.
        malformed_rate: Per-frame chance of a malformed message.
        late_arrival_rate: Per-frame chance of a late arrival (order swap within 3 frames).
    """

    noise_sigma_cm: float = 0.2
    ts_jitter_ms: float = 1.0
    drop_burst_rate: float = 0.002
    duplicate_rate: float = 0.001
    sparse_rate: float = 0.001
    jump_rate: float = 0.0002
    malformed_rate: float = 0.0005
    late_arrival_rate: float = 0.001

    @classmethod
    def clean(cls) -> "SimConfig":
        """Build a config with no Gaussian noise, jitter, or faults.

        Dithering of the cloud points to the cm grid remains, so the clean positions are
        exact only up to an unbiased rounding of about 0.03 cm on the centroid.

        Returns:
            A `SimConfig` with every field at zero.
        """
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class DriveMeta:
    """Metadata of one simulated drive.

    Attributes:
        drive_id: Drive identifier such as ``drive-001``.
        vehicle_id: Identifier of the vehicle.
        vehicle_kind: Vehicle kind (a key of ``VEHICLES``).
        profile: Driving profile (a key of ``PROFILE_LIMITS``).
        drive_start_utc: ISO-8601 start time.
        n_frames: Nominal number of frames before faults.
        seed: Random seed used for the drive.
    """

    drive_id: str
    vehicle_id: str
    vehicle_kind: str
    profile: str
    drive_start_utc: str
    n_frames: int
    seed: int


@dataclass
class SimulatedDrive:
    """One simulated drive with its messages and ground-truth fault labels.

    Attributes:
        frames: Frame messages in emission order.
        meta: Drive metadata.
        labels: Ground-truth labels of the injected faults.
    """

    frames: list[dict[str, Any]]
    meta: DriveMeta
    labels: list[FaultLabel]


def simulate_drive(
    drive_id: str,
    profile: str = "scripted",
    vehicle_kind: str = "car",
    n_frames: int = 1000,
    seed: int = 0,
    sim: SimConfig = SimConfig(),
    faults: tuple[Fault, ...] = (),
    cfg: Config = DEFAULT,
) -> SimulatedDrive:
    """Simulate one drive.

    Positions are sampled from the track at each frame's jittered ``ts`` (the stamped ``ts``
    is the capture time), so timestamp jitter never corrupts velocity estimates.

    Every malformed message also yields a ``gap_after_malformed`` label (VL-001) on the next
    frame, where the pipeline sees a ``frame_idx`` gap.

    Args:
        drive_id: Identifier written into every message.
        profile: A key of ``PROFILE_LIMITS``.
        vehicle_kind: A key of ``VEHICLES``.
        n_frames: Nominal number of frames.
        seed: Random seed; the same arguments give an identical drive.
        sim: Noise and random-fault rates.
        faults: Explicit faults applied in addition to the random ones.
        cfg: Pipeline thresholds; ``cfg.frame_period_s`` is the time step.

    Returns:
        The simulated drive with frames, metadata and labels.

    Raises:
        KeyError: If ``vehicle_kind`` is not a key of ``VEHICLES``.
        ValueError: If ``profile`` is not a key of ``PROFILE_LIMITS``, or the explicit
            ``faults`` collide (see `apply_faults`).
    """
    # One independent stream per concern, so e.g. the late-arrival rate cannot shift the noise.
    noise, jitter, fault_rng, late, meta_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(5)
    )
    meta = _make_meta(drive_id, profile, vehicle_kind, n_frames, seed, meta_rng)
    track = generate_track(profile, n_frames, seed, cfg)
    ts = _timestamps(n_frames, sim, jitter, cfg.frame_period_s)
    frames = _render_frames(meta, track, ts, sim, noise, cfg.frame_period_s)
    all_faults = (*faults, *random_faults(n_frames, sim, fault_rng))
    faulty, labels = apply_faults(frames, all_faults, drive_id)
    labels = [*labels, *malformed_gap_labels(faulty, labels)]
    return SimulatedDrive(_swap_late(faulty, sim.late_arrival_rate, late), meta, labels)


def _make_meta(
    drive_id: str,
    profile: str,
    vehicle_kind: str,
    n_frames: int,
    seed: int,
    rng: np.random.Generator,
) -> DriveMeta:
    """Build the metadata of a drive; the start time is drawn from the drive's seed.

    Args:
        drive_id: Drive identifier.
        profile: Driving profile.
        vehicle_kind: Vehicle kind.
        n_frames: Nominal number of frames.
        seed: Drive seed.
        rng: Random generator of the metadata stream.

    Returns:
        The metadata; ``drive_start_utc`` lies within one day after 2026-01-01T00:00Z.
    """
    start = _EPOCH + timedelta(seconds=int(rng.integers(0, _DAY_S)))
    return DriveMeta(
        drive_id, f"veh-{drive_id}", vehicle_kind, profile, start.isoformat(), n_frames, seed
    )


def _timestamps(n_frames: int, sim: SimConfig, rng: np.random.Generator, dt: float) -> Floats:
    """Nominal frame times plus clipped Gaussian jitter.

    Args:
        n_frames: Number of frames.
        sim: Simulator config holding ``ts_jitter_ms``.
        rng: Random generator of the jitter stream.
        dt: Frame period in seconds.

    Returns:
        The stamped capture time of every frame, in seconds.
    """
    jitter_ms = rng.normal(0.0, sim.ts_jitter_ms, n_frames)
    return np.arange(n_frames) * dt + np.clip(jitter_ms, -_JITTER_CLIP_MS, _JITTER_CLIP_MS) / 1000.0


def _sample(values: Floats, ts: Floats, dt: float) -> Floats:
    """Sample a per-frame signal at arbitrary times, extrapolating linearly past the ends.

    Args:
        values: One value per nominal frame time ``k * dt``.
        ts: The times to sample at, in seconds.
        dt: Frame period in seconds.

    Returns:
        The linearly interpolated values; before the first and after the last frame the
        end slope continues, so jitter at the drive edges adds no position error either.
    """
    if len(values) < 2:
        return np.full_like(ts, values[0])
    times = np.arange(len(values)) * dt
    head = np.minimum(ts, 0.0) * (values[1] - values[0]) / dt
    tail = np.maximum(ts - times[-1], 0.0) * (values[-1] - values[-2]) / dt
    sampled: Floats = np.interp(ts, times, values) + head + tail
    return sampled


def _stochastic_round(values: Floats, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """Round each value to an integer as ``floor(v) + Bernoulli(frac(v))``.

    Unbiased, so the mean of many rounded values keeps the sub-integer part of the signal.

    Args:
        values: Values to round.
        rng: Random generator of the noise stream.

    Returns:
        The rounded integers.
    """
    low = np.floor(values)
    return (low + (rng.random(values.shape) < values - low)).astype(np.int64)


def _cloud(
    lattice: npt.NDArray[np.int64], center_cm: Floats, rng: np.random.Generator
) -> list[list[int]]:
    """Place the vehicle lattice at a position, dithering every point to the cm grid.

    Args:
        lattice: Integer-cm ``(n, 3)`` offsets of the vehicle's points.
        center_cm: The ``x, y`` position of the vehicle in centimeters.
        rng: Random generator of the noise stream.

    Returns:
        The points as a list of ``[x_cm, y_cm, z_cm]`` integer lists.
    """
    xy = _stochastic_round(lattice[:, :2] + center_cm, rng)
    return cast(list[list[int]], np.column_stack([xy, lattice[:, 2]]).tolist())


def _render_frames(
    meta: DriveMeta,
    track: Track,
    ts: Floats,
    sim: SimConfig,
    rng: np.random.Generator,
    dt: float,
) -> list[dict[str, Any]]:
    """Turn a track into clean frame messages (noise and jitter, no faults).

    Positions are sampled from the track at the jittered ``ts``; Gaussian position noise is
    added per axis before the per-point dithering.

    Args:
        meta: Drive metadata copied into every message.
        track: The ground-truth track.
        ts: Stamped capture time of every frame.
        sim: Simulator config holding ``noise_sigma_cm``.
        rng: Random generator of the noise stream.
        dt: Frame period in seconds.

    Returns:
        One message per frame, in frame order.
    """
    truth = 100.0 * np.stack([_sample(track.x, ts, dt), _sample(track.y, ts, dt)], axis=1)
    centers = truth + rng.normal(0.0, sim.noise_sigma_cm, truth.shape)
    lattice = cloud_lattice(VEHICLES[meta.vehicle_kind])
    return [
        {
            "drive_id": meta.drive_id,
            "vehicle_id": meta.vehicle_id,
            "vehicle_kind": meta.vehicle_kind,
            "drive_start_utc": meta.drive_start_utc,
            "frame_idx": k,
            "ts": float(ts[k]),
            "points": _cloud(lattice, centers[k], rng),
        }
        for k in range(len(ts))
    ]


def _swap_late(
    frames: list[dict[str, Any]], rate: float, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Emit some frames late: swap each chosen frame with one up to two positions ahead.

    Args:
        frames: Frames in emission order.
        rate: Per-frame chance of being late.
        rng: Random generator of the late-arrival stream.

    Returns:
        The reordered frames; content, ``frame_idx`` and ``ts`` are unchanged.
    """
    out = list(frames)
    late = np.flatnonzero(rng.random(len(out)) < rate)
    reach = rng.integers(1, _LATE_REACH + 1, size=len(out))
    for i in late:
        j = min(int(i) + int(reach[i]), len(out) - 1)
        out[i], out[j] = out[j], out[i]
    return out


def simulate_fleet(
    n_drives: int,
    seed: int = 0,
    n_frames: int = 1000,
    sim: SimConfig = SimConfig(),
    faults: Mapping[int, tuple[Fault, ...]] | None = None,
) -> list[SimulatedDrive]:
    """Simulate a fleet of drives with ids ``drive-001``, ``drive-002``, ...

    Profiles (highway, city, curvy, parking) and vehicle kinds (car, suv, truck) cycle
    deterministically; start times are spread over one day.

    Args:
        n_drives: Number of drives.
        seed: Fleet seed; the same arguments give an identical fleet.
        n_frames: Nominal number of frames per drive.
        sim: Noise and random-fault rates.
        faults: Explicit faults by zero-based drive index (``0`` is ``drive-001``), applied in
            addition to the random ones; ``None`` adds none.

    Returns:
        The simulated drives in id order.
    """
    explicit = faults or {}
    children = np.random.SeedSequence(seed).spawn(n_drives)
    return [
        simulate_drive(
            f"drive-{i + 1:03d}",
            _FLEET_PROFILES[i % len(_FLEET_PROFILES)],
            _FLEET_VEHICLES[i % len(_FLEET_VEHICLES)],
            n_frames,
            int(child.generate_state(1)[0]),
            sim,
            explicit.get(i, ()),
        )
        for i, child in enumerate(children)
    ]
