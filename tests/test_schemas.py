"""Tests for the table schemas: DDL strings and column lists must stay in step."""

from vl import schemas

DDL_TO_COLUMNS = [
    ("raw_frames", schemas.RAW_FRAMES_DDL, schemas.RAW_FRAMES_COLUMNS),
    ("frames", schemas.FRAMES_DDL, schemas.FRAMES_COLUMNS),
    ("events", schemas.EVENTS_DDL, schemas.EVENTS_COLUMNS),
    ("drives", schemas.DRIVES_DDL, schemas.DRIVES_COLUMNS),
    ("rejects", schemas.REJECTS_DDL, schemas.REJECTS_COLUMNS),
]


def test_columns_match_ddl_order() -> None:
    for table, ddl, columns in DDL_TO_COLUMNS:
        assert [part.split()[0] for part in ddl.split(", ")] == columns, table


def test_frames_dtypes_cover_every_frames_column() -> None:
    assert list(schemas.FRAMES_DTYPES) == schemas.FRAMES_COLUMNS
