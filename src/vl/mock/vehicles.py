"""Vehicle shapes and the point-cloud lattice that stands in for a vehicle's sensor cloud.

Role in the pipeline: the simulator places one lattice per frame at the vehicle position.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class VehicleSpec:
    """Bounding-box dimensions of one vehicle kind.

    Attributes:
        kind: Vehicle kind name (``car``, ``suv`` or ``truck``).
        length_m: Length in meters (x axis).
        width_m: Width in meters (y axis).
        height_m: Height in meters (z axis).
    """

    kind: str
    length_m: float
    width_m: float
    height_m: float


VEHICLES: dict[str, VehicleSpec] = {
    "car": VehicleSpec("car", 4.5, 1.8, 1.5),
    "suv": VehicleSpec("suv", 4.8, 1.9, 1.8),
    "truck": VehicleSpec("truck", 8.0, 2.5, 3.2),
}


def cloud_lattice(spec: VehicleSpec, pitch_cm: int = 50) -> npt.NDArray[np.int64]:
    """Build the integer-cm point lattice of a vehicle, centered on the origin.

    Each axis has ``round(size / pitch) + 1`` points, so a car has 10 x 5 x 4 = 200 points.

    Args:
        spec: The vehicle whose box is sampled.
        pitch_cm: Spacing between neighboring lattice points, in centimeters.

    Returns:
        An ``(n_points, 3)`` integer array of ``x, y, z`` offsets in centimeters.
    """
    sizes = (spec.length_m, spec.width_m, spec.height_m)
    axes = [_axis_cm(size, pitch_cm) for size in sizes]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1)


def _axis_cm(size_m: float, pitch_cm: int) -> npt.NDArray[np.int64]:
    """Build the symmetric integer-cm coordinates of one lattice axis.

    Args:
        size_m: Box size along the axis, in meters.
        pitch_cm: Target spacing between neighboring points, in centimeters.

    Returns:
        ``round(size / pitch) + 1`` integer coordinates, symmetric around zero.
    """
    size_cm = size_m * 100.0
    count = round(size_cm / pitch_cm) + 1
    return np.rint(np.linspace(-size_cm / 2.0, size_cm / 2.0, count)).astype(np.int64)
