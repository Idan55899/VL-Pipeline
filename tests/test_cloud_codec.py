"""Tests for the rigid-body cloud codec (`vl.cloud_codec`)."""

import json

import numpy as np

from vl.cloud_codec import clouds_of, decode_clouds, encode_clouds, max_residuals
from vl.mock.drive_sim import SimConfig, simulate_drive
from vl.mock.faults import Fault


def test_round_trip_is_lossless_for_every_kind_of_frame() -> None:
    """Rigid frames, a sparse and a duplicate frame, a malformed (None) one, huge coordinates."""
    faults = (Fault("sparse", 50), Fault("duplicate", 80), Fault("malformed", 120))
    drive = simulate_drive("d", "highway", "truck", 300, 1, SimConfig.clean(), faults)
    clouds = clouds_of(drive.frames)
    far = clouds[10] + np.array([3_000_000_000, -2_000_000_000, 7])  # beyond 32 bits, still exact
    deformed = clouds[11].copy()
    deformed[0] += 500  # one point 5 m off: the frame is stored raw
    clouds[10], clouds[11] = far, deformed
    decoded = decode_clouds(encode_clouds(clouds))
    assert len(decoded) == len(clouds) and sum(c is None for c in decoded) == 1
    for got, want in zip(decoded, clouds, strict=True):
        assert (got is None and want is None) or np.array_equal(got, want)
    residuals = max_residuals(clouds)
    assert residuals[11] is None and residuals[12] is not None and residuals[12] <= 2


def test_compresses_a_realistic_drive_far_below_json() -> None:
    drive = simulate_drive("d", "city", "car", 1000, 0, SimConfig())  # noise and jitter on
    clouds = clouds_of(drive.frames)
    texts = [json.dumps(c.tolist(), separators=(",", ":")) for c in clouds if c is not None]
    assert len(encode_clouds(clouds)) * 10 < sum(len(text) for text in texts)
