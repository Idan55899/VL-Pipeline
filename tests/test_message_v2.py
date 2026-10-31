"""Message v2 (claim check): both strict parsers accept it and reject the same bad v2 lines.

A v2 message carries the centroid in meters, the point count and a reference to the drive's cloud
file instead of the points. These tests pin that the validator (pure Python) and the Spark ingest
give the same raw row for a v1 message and its v2 twin, and both reject every malformed v2 line.
"""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest
from pyspark.sql import SparkSession

from ingest_helpers import raw_df
from vl.etl.ingest import parse_messages
from vl.validate import _read_jsonl_source

_META = {"drive_id": "d", "vehicle_kind": "car", "frame_idx": 7, "ts": 0.4375}
V1 = {**_META, "points": [[100, 200, 300], [103, 196, 301]]}
V2 = {
    **_META,
    "schema_version": 2,
    "centroid_m": [1.015, 1.98, 3.005],
    "n_points": 2,
    "cloud_ref": "clouds/d.vlc#0",
}
V2_EMPTY = {**_META, "frame_idx": 8, "schema_version": 2, "n_points": 0, "cloud_ref": "c#1"}
VALID = [V1, {**V1, "frame_idx": 9, "schema_version": 1}, V2, V2_EMPTY]


def _without(message: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a copy of the message without one key."""
    return {k: v for k, v in message.items() if k != key}


INVALID = {
    "version_string": {**V2, "schema_version": "2"},
    "version_float": {**V2, "schema_version": 2.0},
    "version_unknown": {**V2, "schema_version": 3},
    "version_bool": {**V2, "schema_version": True},
    "no_cloud_ref": _without(V2, "cloud_ref"),
    "cloud_ref_number": {**V2, "cloud_ref": 5},
    "no_centroid": _without(V2, "centroid_m"),
    "centroid_two_values": {**V2, "centroid_m": [1.0, 2.0]},
    "centroid_string_value": {**V2, "centroid_m": [1.0, "2", 3.0]},
    "centroid_bool_value": {**V2, "centroid_m": [1.0, True, 3.0]},
    "centroid_null_value": {**V2, "centroid_m": [1.0, None, 3.0]},
    "centroid_on_empty_cloud": {**V2_EMPTY, "centroid_m": [1.0, 2.0, 3.0]},
    "centroid_string_on_empty_cloud": {**V2_EMPTY, "centroid_m": "malformed"},
    "n_points_negative": {**V2, "n_points": -1},
    "n_points_float": {**V2, "n_points": 2.0},
    "n_points_beyond_int32": {**V2, "n_points": 2**31},
    "no_n_points": _without(V2, "n_points"),
    "v1_without_points": _without({**V2, "schema_version": None}, "schema_version"),
}
EXPECTED = [  # (frame_idx, n_points, x, y, z) of the valid lines, sorted
    (7, 2, 1.015, 1.98, 3.005),
    (7, 2, 1.015, 1.98, 3.005),
    (8, 0, None, None, None),
    (9, 2, 1.015, 1.98, 3.005),
]


def _rows(table: Iterable[Sequence[object]]) -> list[tuple[object, ...]]:
    """Reduce raw rows to ``(frame_idx, n_points, x, y, z)``, centroids rounded, NaN as None."""

    def num(value: object) -> float | None:
        number = None if value is None else float(value)  # type: ignore[arg-type]
        return None if number is None or number != number else round(number, 9)

    rows = [(r[0], r[1], num(r[2]), num(r[3]), num(r[4])) for r in table]
    return sorted(rows, key=lambda row: (str(row[0]), str(row[1])))


@pytest.mark.parametrize("name", INVALID)
def test_validator_rejects_malformed_v2(name: str, tmp_path: Path) -> None:
    path = tmp_path / "frames.jsonl"
    path.write_text(json.dumps(INVALID[name]) + "\n")
    assert _read_jsonl_source(path).n_malformed == 1


def test_validator_reads_v2_like_its_v1_twin(tmp_path: Path) -> None:
    path = tmp_path / "frames.jsonl"
    path.write_text("".join(json.dumps(m) + "\n" for m in VALID))
    raw = _read_jsonl_source(path).raw
    columns = ["frame_idx", "n_points", "x", "y", "z"]
    assert _rows(raw[columns].itertuples(index=False)) == EXPECTED


def test_spark_ingest_agrees_on_v2(spark: SparkSession) -> None:
    lines = [json.dumps(m) for m in VALID + list(INVALID.values())]
    valid, rejects = parse_messages(raw_df(spark, lines))
    got = valid.select("frame_idx", "n_points", "x", "y", "z").collect()
    assert _rows(got) == EXPECTED
    assert rejects.count() == len(INVALID)
