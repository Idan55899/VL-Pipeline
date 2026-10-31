"""Kafka track unit tests: wire format, publishing, demo fleet, producer CLI, ingest args, compose.

No Docker, broker or network: the producer is a fake, time is a fake clock, and the compose
file is checked as text.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from kafka_helpers import FakeClock, FakeProducer
from pipeline_helpers import raw_row
from sim_helpers import make_frames
from vl.claim_check import load_cloud
from vl.etl.features import derive_frames, detect_events
from vl.etl.ingest import parse_args
from vl.kafka_io import BOOTSTRAP, TOPIC, decode_frame, encode_frame, publish_drive
from vl.mock.producer import build_fleet, main

ROOT = Path(__file__).resolve().parent.parent
DEMO_KINDS = {"drop", "jump", "duplicate", "malformed"}
GAP_KIND = "gap_after_malformed"


def test_encode_decode_roundtrip_and_errors() -> None:
    """Frames survive the wire (also malformed-looking and unicode); bad bytes raise."""
    frames: list[dict[str, object]] = [
        {"drive_id": "d", "frame_idx": 3, "ts": 0.1875, "points": [[1, -2, 3]]},
        {"drive_id": "d", "frame_idx": 4, "ts": 0.25, "points": "garbage"},
        {"drive_id": "drive-é", "vehicle_id": "véh-ü", "points": []},
    ]
    for frame in frames:
        raw = encode_frame(frame)
        assert isinstance(raw, bytes) and decode_frame(raw) == frame, frame
    assert encode_frame({"a": 1, "b": "é"}) == '{"a":1,"b":"é"}'.encode("utf-8")
    for bad in (b"not json", b"", b"[1,2]", b'"text"', b"7", b"\xff\xfe", b'{"a":1} x'):
        with pytest.raises(ValueError):
            decode_frame(bad)


def test_publish_drive_key_order_count_and_flush() -> None:
    """Each message is keyed by the drive id, in order, malformed frames unchanged."""
    frames: list[dict[str, object]] = [*make_frames(4, "drive-007")]
    frames.insert(2, {**frames[1], "frame_idx": 9, "points": "garbage"})
    producer = FakeProducer()
    assert publish_drive(frames, "unused:1", "t-x", producer=producer) == len(frames)
    assert [topic for topic, _, _ in producer.sent] == ["t-x"] * len(frames)
    assert {key for _, key, _ in producer.sent} == {b"drive-007"}
    assert [decode_frame(value) for _, _, value in producer.sent] == frames
    assert producer.flushes == 1


def test_publish_drive_pacing() -> None:
    """With a rate, message i is sent at i/rate on a fake clock; without one, never sleeps."""
    frames: list[dict[str, object]] = [*make_frames(5)]
    paced = FakeClock()
    producer = FakeProducer(paced)
    publish_drive(frames, producer=producer, rate_fps=4.0, sleep=paced.sleep, clock=paced.clock)
    assert producer.times == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert sum(paced.sleeps) == pytest.approx(1.0) and min(paced.sleeps) > 0
    free = FakeClock()
    publish_drive(frames, producer=FakeProducer(), sleep=free.sleep, clock=free.clock)
    assert free.sleeps == []


def test_cli_cloud_dir_sends_small_v2_messages_whose_clouds_round_trip(tmp_path: Path) -> None:
    """With --cloud-dir every message is v2 (no points) and its cloud_ref returns the cloud."""
    argv = ["--drives", "2", "--frames", "100", "--seed", "1", "--faults", "demo"]
    v1, v2 = FakeProducer(), FakeProducer()
    assert main(argv, producer=v1) == 0
    assert main([*argv, "--cloud-dir", str(tmp_path / "clouds")], producer=v2) == 0
    old = [decode_frame(value) for _, _, value in v1.sent]
    new = [decode_frame(value) for _, _, value in v2.sent]
    assert len(new) == len(old) and all(m["schema_version"] == 2 and "points" not in m for m in new)
    assert sum(len(v) for _, _, v in v2.sent) * 4 < sum(len(v) for _, _, v in v1.sent)
    for before, after in zip(old, new, strict=True):
        cloud = load_cloud(str(after["cloud_ref"]))
        if before["points"] == "garbage":
            assert cloud is None and after["centroid_m"] == "malformed"
        else:
            assert cloud is not None and cloud.tolist() == before["points"]


def test_build_fleet_scripted_appends_a_clean_scenario_drive() -> None:
    """``scripted=True`` adds ``drive-004`` with HARD_BRAKE, SHARP_TURN and STOP and no faults."""
    fleet = build_fleet(3, 1000, 0, "demo", scripted=True)
    drive = fleet[-1]
    assert [d.meta.drive_id for d in fleet] == ["drive-001", "drive-002", "drive-003", "drive-004"]
    assert (drive.meta.profile, drive.labels) == ("scripted", [])
    events = detect_events(derive_frames(pd.DataFrame(raw_row(f) for f in drive.frames)))
    assert {"HARD_BRAKE", "SHARP_TURN", "STOP"} <= set(events["event_type"])
    assert len(build_fleet(3, 1000, 0, "demo")) == 3, "scripted is off by default"


def test_build_fleet_demo_faults_are_explicit_and_deterministic() -> None:
    """Demo puts exactly drop, jump, duplicate, malformed on drive-002; none puts nothing."""
    fleet = build_fleet(3, 200, 5, "demo")
    labels = {d.meta.drive_id: d.labels for d in fleet}
    kinds = Counter(label.kind for label in labels["drive-002"])
    assert kinds == Counter({"drop": 1, "jump": 1, "duplicate": 1, "malformed": 1, GAP_KIND: 1})
    assert labels["drive-001"] == [] and labels["drive-003"] == []
    spots = sorted(x.frame_idx for x in labels["drive-002"] if x.kind != GAP_KIND)
    assert all(1 <= s <= 198 for s in spots)
    assert all(b - a >= 10 for a, b in zip(spots, spots[1:])), spots
    assert [len(d.frames) for d in fleet] == [200, 200, 200]  # a drop removes, a duplicate adds
    again = build_fleet(3, 200, 5, "demo")
    assert [d.frames for d in again] == [d.frames for d in fleet]
    assert build_fleet(3, 200, 6, "demo")[0].frames != fleet[0].frames
    assert all(d.labels == [] for d in build_fleet(3, 200, 5, "none")), "faults=none"


def check_fleet_files(out: Path) -> list[dict[str, object]]:
    """Check frames.jsonl and labels.json of the demo run; return the written messages."""
    lines = (out / "frames.jsonl").read_text().splitlines()
    assert len(lines) == 300  # per drive: 100 frames, the drop and the duplicate cancel out
    labels = json.loads((out / "labels.json").read_text())
    assert len(labels["drives"]) == 3
    assert {x["kind"] for x in labels["labels"]} == DEMO_KINDS | {GAP_KIND}
    return [json.loads(line) for line in lines]


def test_producer_main_publishes_and_writes_fleet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI sends every message in order, writes the fleet files, prints one summary."""
    producer, out = FakeProducer(), tmp_path / "mock"
    argv = ["--drives", "3", "--frames", "100", "--seed", "1", "--faults", "demo"]
    assert main([*argv, "--out-dir", str(out), "--topic", "t-y"], producer=producer) == 0
    written = check_fleet_files(out)
    sent = [decode_frame(value) for _, _, value in producer.sent]
    assert sent == written
    assert {topic for topic, _, _ in producer.sent} == {"t-y"}
    assert [key for _, key, _ in producer.sent] == [str(m["drive_id"]).encode() for m in sent]
    assert sum(m["points"] == "garbage" for m in sent) == 1
    summary = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(summary) == 1
    assert re.search(r"\b300\b", summary[0]) and re.search(r"\b3\b", summary[0]), summary


def test_ingest_parse_args() -> None:
    """File and Kafka sources, defaults, mutual exclusion, and exit 2 on bad arguments."""
    files = parse_args(["--source-dir", "in"])
    assert (files.source_dir, files.kafka, files.warehouse) == ("in", False, "data/warehouse")
    assert files.checkpoint == "data/warehouse/_checkpoints/ingest"
    kafka = parse_args(["--kafka", "--warehouse", "wh"])
    assert (kafka.kafka, kafka.source_dir) == (True, None)
    assert kafka.checkpoint == "wh/_checkpoints/ingest"
    assert (kafka.bootstrap, kafka.topic) == (BOOTSTRAP, TOPIC) and TOPIC == "vl.frames"
    custom = parse_args(["--kafka", "--bootstrap", "h:1", "--topic", "t", "--checkpoint", "cp"])
    assert (custom.bootstrap, custom.topic, custom.checkpoint) == ("h:1", "t", "cp")
    bad = [[], ["--kafka", "--source-dir", "in"], ["--bogus"], ["--source-dir"], ["--topic", "t"]]
    for argv in bad:
        with pytest.raises(SystemExit) as exc:
            parse_args(argv)
        assert exc.value.code == 2, argv


def setting(text: str, key: str) -> str | None:
    """Return the value of a ``KEY: value`` / ``KEY=value`` line of the compose file, or None."""
    found = re.search(rf"^\s*-?\s*{key}\s*[:=]\s*\"?([^\"\n#]+?)\"?\s*(#.*)?$", text, re.MULTILINE)
    return found.group(1) if found else None


def test_compose_file_declares_a_kraft_broker_with_three_partitions() -> None:
    """Pure-data check of docker-compose.yml (text; no Docker needed)."""
    text = (ROOT / "docker-compose.yml").read_text()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert re.search(r"^\s*image:\s*apache/kafka:\S+", code, re.MULTILINE)
    assert "zookeeper" not in code.lower()
    assert setting(code, "KAFKA_PROCESS_ROLES") == "broker,controller"
    assert setting(code, "KAFKA_NUM_PARTITIONS") == "3"
    assert setting(code, "KAFKA_AUTO_CREATE_TOPICS_ENABLE") == "true"
    assert setting(code, "KAFKA_ADVERTISED_LISTENERS") == "PLAINTEXT://localhost:9092"
    assert re.search(r"9092:9092", code)
