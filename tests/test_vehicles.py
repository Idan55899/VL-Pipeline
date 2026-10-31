"""Tests for the vehicle table and the point-cloud lattice."""

import numpy as np

from vl.mock.vehicles import VEHICLES, VehicleSpec, cloud_lattice


def test_vehicle_point_counts() -> None:
    for kind, count in [("car", 200), ("suv", 275), ("truck", 714)]:
        points = cloud_lattice(VEHICLES[kind])
        assert points.shape == (count, 3), kind
        assert np.issubdtype(points.dtype, np.integer), kind
        assert np.all(np.abs(points.mean(axis=0)) <= 1.0), kind


def test_vehicles_table() -> None:
    assert VEHICLES == {
        "car": VehicleSpec("car", 4.5, 1.8, 1.5),
        "suv": VehicleSpec("suv", 4.8, 1.9, 1.8),
        "truck": VehicleSpec("truck", 8.0, 2.5, 3.2),
    }
