"""Error and event vocabularies shared by every stage of the pipeline.

`ErrorCode` labels frames that fail validation (one code per frame, see the spec for
precedence); `EventType` names the driving events detected on valid frames.
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Per-frame validation error code with a human-readable description.

    Attributes:
        description: What the code means, for reports and the dashboard.
    """

    description: str

    def __new__(cls, code: str, description: str) -> "ErrorCode":
        """Create a member whose value is the code string.

        Args:
            code: The code string, e.g. ``"VL-001"``.
            description: Human-readable meaning of the code.

        Returns:
            The new enum member.
        """
        member = str.__new__(cls, code)
        member._value_ = code
        member.description = description
        return member

    DROPPED_FRAME = ("VL-001", "Dropped frame: gap in frame_idx")
    POSITION_JUMP = ("VL-002", "Position jump: implied displacement beyond the limit")
    IMPOSSIBLE_ACCEL = ("VL-003", "Impossible acceleration: beyond physical limits")
    DUPLICATE_FRAME = ("VL-004", "Duplicate or out-of-order frame")
    MALFORMED_MESSAGE = ("VL-005", "Malformed message: cannot be parsed")
    SPARSE_CLOUD = ("VL-006", "Empty or sparse point cloud")


class EventType(str, Enum):
    """Driving event detected on a valid drive."""

    HARD_BRAKE = "HARD_BRAKE"
    HARD_ACCEL = "HARD_ACCEL"
    SHARP_TURN = "SHARP_TURN"
    STOP = "STOP"
