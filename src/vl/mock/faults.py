"""Fault injection: explicit and random faults applied to frame messages, with labels.

Role in the pipeline: turns clean simulated frames into the dirty stream the validator must
cope with, and records which error code each fault is expected to trigger.
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from vl.errors import ErrorCode

if TYPE_CHECKING:
    from vl.mock.drive_sim import SimConfig

Frame = dict[str, Any]

JUMP_CM = 500  # a position jump shifts the frame and all later ones by 5 m along x
SPARSE_POINTS = 10  # a sparse cloud keeps only this many points
TS_BACKWARDS_S = 0.001  # a backwards timestamp is the previous one minus this
MAX_DROP_FRAMES = 5  # random drop bursts are 1 to this many frames long
_GAP_CODE = ErrorCode.DROPPED_FRAME.value

GAP_AFTER_MALFORMED = "gap_after_malformed"  # label kind, not a `Fault` kind
_NEEDS_PREDECESSOR = ("drop", "jump", "ts_backwards")  # invisible on the first frame


@dataclass(frozen=True)
class Fault:
    """One fault to inject.

    Attributes:
        kind: One of `FAULT_KINDS`: ``drop``, ``jump``, ``duplicate``, ``sparse``,
            ``malformed``, ``ts_backwards``.
        frame_idx: Frame index the fault starts at.
        length: Number of consecutive frames affected (only ``drop`` uses more than 1).
    """

    kind: str
    frame_idx: int
    length: int = 1


@dataclass
class FaultLabel:
    """Ground-truth label of an injected fault.

    Attributes:
        drive_id: Drive the fault belongs to.
        frame_idx: Frame the validator is expected to flag.
        kind: The fault kind, or ``gap_after_malformed`` for the label on the frame after a
            malformed message (the pipeline sees a ``frame_idx`` gap there, VL-001).
        expected_code: Expected error code, or ``None`` for malformed messages (which land
            in ``rejects`` as VL-005 and never in ``frames``).
    """

    drive_id: str
    frame_idx: int
    kind: str
    expected_code: str | None


def _draw_kind(kind: str, rate: float, n_frames: int, rng: np.random.Generator) -> list[Fault]:
    """Draw the faults of one kind, each frame starting one with the kind's rate.

    Args:
        kind: A random fault kind.
        rate: Per-frame chance of the fault starting at a frame.
        n_frames: Number of frames in the drive.
        rng: Random generator.

    Returns:
        The drawn faults in frame order (``drop`` bursts get a random length).
    """
    starts = np.flatnonzero(rng.random(n_frames) < rate)
    if kind == "drop":
        lengths = rng.integers(1, MAX_DROP_FRAMES + 1, size=len(starts))
        return [Fault(kind, int(i), int(n)) for i, n in zip(starts, lengths, strict=True)]
    return [Fault(kind, int(i)) for i in starts]


def _detectable(fault: Fault, n_frames: int) -> bool:
    """Tell whether the validator can see the fault.

    A jump or drop needs a frame before it to be measured against, and a drop needs a frame
    after the gap to carry its label.

    Args:
        fault: The candidate fault.
        n_frames: Number of frames in the drive.

    Returns:
        ``True`` when the fault leaves an observable trace.
    """
    if fault.kind in _NEEDS_PREDECESSOR and fault.frame_idx == 0:
        return False
    return fault.frame_idx + (fault.length if fault.kind == "drop" else 0) < n_frames


def _claimed(fault: Fault) -> range:
    """Frames a fault occupies: its own, plus the frame that carries its consequence.

    A drop occupies its gap and the first frame after it (labelled VL-001); a malformed
    message occupies its frame and the next one (labelled as a gap); other faults one frame.

    Args:
        fault: The fault.

    Returns:
        The contiguous range of ``frame_idx`` values the fault occupies.
    """
    own = fault.length if fault.kind == "drop" else 1
    consequence = 1 if fault.kind in ("drop", "malformed") else 0
    return range(fault.frame_idx, fault.frame_idx + own + consequence)


def _separate(faults: list[Fault]) -> tuple[Fault, ...]:
    """Keep faults apart so no two occupy the same frame (see `_claimed`).

    Args:
        faults: Candidate faults, sorted by ``frame_idx``.

    Returns:
        The kept faults.
    """
    kept: list[Fault] = []
    free_from = 0
    for fault in faults:
        if fault.frame_idx >= free_from:
            kept.append(fault)
            free_from = _claimed(fault).stop
    return tuple(kept)


def random_faults(n_frames: int, sim: "SimConfig", rng: np.random.Generator) -> tuple[Fault, ...]:
    """Draw random faults for one drive from the per-frame rates in ``sim``.

    Faults that the validator could not see (a jump or drop at frame 0, a drop running past
    the end) or that would land on a frame another fault occupies (including the frame after
    a drop or a malformed message) are discarded.

    Args:
        n_frames: Number of frames in the drive.
        sim: Simulator config holding the fault rates.
        rng: Random generator; the same state gives the same faults.

    Returns:
        The drawn faults sorted by frame (``drop`` bursts have length 1 to 5; ``ts_backwards``
        is never drawn).
    """
    rated = [(kind, spec.rate_field) for kind, spec in _SPECS.items() if spec.rate_field]
    drawn = [
        fault
        for kind, field in rated
        for fault in _draw_kind(kind, getattr(sim, field), n_frames, rng)
    ]
    visible = [fault for fault in drawn if _detectable(fault, n_frames)]
    return _separate(sorted(visible, key=lambda fault: fault.frame_idx))


def _find(frames: list[Frame], frame_idx: int) -> int | None:
    """Find the first emitted position of a frame.

    Args:
        frames: Frames in emission order.
        frame_idx: The ``frame_idx`` to look for.

    Returns:
        The list position, or ``None`` when the frame is not emitted.
    """
    return next((pos for pos, frame in enumerate(frames) if frame["frame_idx"] == frame_idx), None)


def _label(drive_id: str, frame_idx: int, fault: Fault) -> FaultLabel:
    """Build the label of a fault.

    Args:
        drive_id: Drive the fault belongs to.
        frame_idx: Frame the validator is expected to flag.
        fault: The applied fault.

    Returns:
        The label with the fault kind's expected code.
    """
    return FaultLabel(drive_id, frame_idx, fault.kind, _SPECS[fault.kind].code)


def _apply_drop(
    frames: list[Frame], fault: Fault, drive_id: str
) -> tuple[list[Frame], list[FaultLabel]]:
    """Omit ``fault.length`` consecutive frames; label the first frame after the gap.

    Args:
        frames: Frames in emission order.
        fault: The drop fault.
        drive_id: Drive id for the label.

    Returns:
        The frames without the gap and the label (none when nothing was dropped or no frame
        follows the gap).
    """
    gap = range(fault.frame_idx, fault.frame_idx + fault.length)
    kept = [frame for frame in frames if frame["frame_idx"] not in gap]
    after = fault.frame_idx + fault.length
    labelled = len(kept) < len(frames) and _find(kept, after) is not None
    return kept, [_label(drive_id, after, fault)] if labelled else []


def _apply_jump(
    frames: list[Frame], fault: Fault, drive_id: str
) -> tuple[list[Frame], list[FaultLabel]]:
    """Shift the frame and all later ones by 5 m along x.

    Args:
        frames: Frames in emission order.
        fault: The jump fault.
        drive_id: Drive id for the label.

    Returns:
        The shifted frames and the label (none when the frame is not emitted).
    """
    if _find(frames, fault.frame_idx) is None:
        return frames, []
    out = [_shift(f) if f["frame_idx"] >= fault.frame_idx else f for f in frames]
    return out, [_label(drive_id, fault.frame_idx, fault)]


def _shift(frame: Frame) -> Frame:
    """Copy a frame with every point moved ``JUMP_CM`` along x.

    Args:
        frame: The frame to shift.

    Returns:
        The shifted copy (a malformed frame, whose points are not a list, is returned as is);
        the input is untouched.
    """
    if not isinstance(frame["points"], list):
        return frame
    return {**frame, "points": [[x + JUMP_CM, y, z] for x, y, z in frame["points"]]}


def _edited(frame: Frame, kind: str, previous: Frame | None) -> Frame | None:
    """Copy a frame with a content fault applied.

    Args:
        frame: The frame to damage.
        kind: ``sparse``, ``malformed`` or ``ts_backwards``.
        previous: The frame emitted just before (needed by ``ts_backwards``).

    Returns:
        The damaged copy, or ``None`` when the fault cannot apply (no previous frame, or the
        frame is already malformed).
    """
    if kind == "malformed":
        return {**frame, "points": "garbage"}
    if not isinstance(frame["points"], list):
        return None
    if kind == "sparse":
        keep = np.linspace(0, len(frame["points"]) - 1, SPARSE_POINTS).round().astype(int)
        return {**frame, "points": [frame["points"][i] for i in keep]}
    if previous is None:
        return None
    return {**frame, "ts": previous["ts"] - TS_BACKWARDS_S}


def _apply_edit(
    frames: list[Frame], fault: Fault, drive_id: str
) -> tuple[list[Frame], list[FaultLabel]]:
    """Apply a content fault (``sparse``, ``malformed`` or ``ts_backwards``) to one frame.

    Args:
        frames: Frames in emission order.
        fault: The fault.
        drive_id: Drive id for the label.

    Returns:
        The frames and the label (none when the frame is not emitted or the fault cannot
        apply, e.g. ``ts_backwards`` on the first frame).
    """
    pos = _find(frames, fault.frame_idx)
    if pos is None:
        return frames, []
    edited = _edited(frames[pos], fault.kind, frames[pos - 1] if pos else None)
    if edited is None:
        return frames, []
    return [*frames[:pos], edited, *frames[pos + 1 :]], [_label(drive_id, fault.frame_idx, fault)]


def _apply_duplicate(
    frames: list[Frame], fault: Fault, drive_id: str
) -> tuple[list[Frame], list[FaultLabel]]:
    """Emit the frame twice in a row.

    Args:
        frames: Frames in emission order.
        fault: The duplicate fault.
        drive_id: Drive id for the label.

    Returns:
        The frames with the copy inserted and the label (none when the frame is not emitted).
    """
    pos = _find(frames, fault.frame_idx)
    if pos is None:
        return frames, []
    return [*frames[: pos + 1], copy.deepcopy(frames[pos]), *frames[pos + 1 :]], [
        _label(drive_id, fault.frame_idx, fault)
    ]


_Applier = Callable[[list[Frame], Fault, str], tuple[list[Frame], list[FaultLabel]]]


class _Spec(NamedTuple):
    """Everything the module knows about one fault kind.

    Attributes:
        code: Expected error code (``None`` for malformed: it lands in rejects as VL-005).
        rate_field: `SimConfig` field with the per-frame rate, ``None`` if never drawn.
        apply: Function injecting the fault.
    """

    code: str | None
    rate_field: str | None
    apply: _Applier


# The single place that enumerates the fault kinds.
_SPECS: dict[str, _Spec] = {
    "drop": _Spec(ErrorCode.DROPPED_FRAME.value, "drop_burst_rate", _apply_drop),
    "duplicate": _Spec(ErrorCode.DUPLICATE_FRAME.value, "duplicate_rate", _apply_duplicate),
    "sparse": _Spec(ErrorCode.SPARSE_CLOUD.value, "sparse_rate", _apply_edit),
    "jump": _Spec(ErrorCode.POSITION_JUMP.value, "jump_rate", _apply_jump),
    "malformed": _Spec(None, "malformed_rate", _apply_edit),
    "ts_backwards": _Spec(ErrorCode.DUPLICATE_FRAME.value, None, _apply_edit),
}
FAULT_KINDS = tuple(_SPECS)


def _check_faults(active: list[Fault]) -> None:
    """Reject faults the validator cannot see or that occupy the same frame.

    Args:
        active: The faults whose frame is emitted, sorted by ``frame_idx``.

    Raises:
        ValueError: For a ``drop``, ``jump`` or ``ts_backwards`` at frame 0, or when two
            faults occupy the same frame (see `_claimed`); the message names the faults.
    """
    for fault in active:
        if fault.kind in _NEEDS_PREDECESSOR and fault.frame_idx == 0:
            raise ValueError(f"{fault} is undetectable on the first frame")
    for first, second in zip(active, active[1:], strict=False):
        if second.frame_idx in _claimed(first):
            raise ValueError(f"{first} collides with {second}: they occupy the same frame")


def apply_faults(
    frames: list[Frame], faults: tuple[Fault, ...], drive_id: str
) -> tuple[list[Frame], list[FaultLabel]]:
    """Apply faults to frame messages and label them.

    Semantics: ``drop`` omits ``length`` frames from ``frame_idx`` (label on the first frame
    after the gap, VL-001); ``jump`` shifts that frame and all later ones by 5 m along x
    (VL-002); ``duplicate`` emits the frame twice (VL-004); ``sparse`` keeps only 10 points
    (VL-006); ``malformed`` replaces ``points`` with ``"garbage"`` (no code); ``ts_backwards``
    sets ``ts`` to the previous frame's ``ts`` minus 0.001 (VL-004).

    Faults are applied in ``frame_idx`` order. A fault whose frame is not in ``frames`` (past
    the end, or already absent) is skipped silently. Each fault returns only its own label;
    the extra label of a malformed message is added by `malformed_gap_labels`. The input
    frames are not modified.

    Args:
        frames: Clean frame messages, in emission order.
        faults: Faults to apply.
        drive_id: Drive id written into every label.

    Returns:
        The faulty frame messages and the fault labels.

    Raises:
        ValueError: If two faults occupy the same frame (the same ``frame_idx``, a drop whose
            gap holds another fault, or a fault on the first frame after a drop or a
            malformed message), or a ``drop``, ``jump`` or ``ts_backwards`` is at frame 0.
    """
    present = {frame["frame_idx"] for frame in frames}
    active = sorted((f for f in faults if f.frame_idx in present), key=lambda f: f.frame_idx)
    _check_faults(active)
    out = list(frames)
    labels: list[FaultLabel] = []
    for fault in active:
        out, new_labels = _SPECS[fault.kind].apply(out, fault, drive_id)
        labels.extend(new_labels)
    return out, labels


def malformed_gap_labels(frames: list[Frame], labels: list[FaultLabel]) -> list[FaultLabel]:
    """Label the frame after each malformed message, where the pipeline sees a gap.

    A malformed message never reaches ``frames``, so the next valid frame follows a
    ``frame_idx`` gap and is expected to get VL-001.

    Args:
        frames: The faulty frames of the drive.
        labels: The fault labels returned by `apply_faults`.

    Returns:
        One ``gap_after_malformed`` label per malformed fault whose next frame is emitted
        (none when the malformed frame was the last one).
    """
    present = {frame["frame_idx"] for frame in frames}
    return [
        FaultLabel(label.drive_id, label.frame_idx + 1, GAP_AFTER_MALFORMED, _GAP_CODE)
        for label in labels
        if label.kind == "malformed" and label.frame_idx + 1 in present
    ]
