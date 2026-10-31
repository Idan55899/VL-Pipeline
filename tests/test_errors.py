"""Tests for vl.errors."""

from vl.errors import ErrorCode


def test_codes() -> None:
    """The error codes are exactly VL-001..VL-006 and each is described."""
    assert {c.value for c in ErrorCode} == {f"VL-00{i}" for i in range(1, 7)}
    assert all(c.description for c in ErrorCode)
