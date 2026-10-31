"""DuckDB query layer over the Parquet warehouse.

Role in the pipeline: the read side. The dashboard, the docs and ad-hoc analysis query the
tables written by the Spark jobs with plain SQL, in process, without a database file.
"""

import glob
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

from vl.errors import EventType
from vl.schemas import TABLE_DDLS

TABLES = tuple(TABLE_DDLS)
# DuckDB type of every Spark type that appears in the DDL strings of `vl.schemas`.
DUCKDB_TYPES = {
    "string": "VARCHAR",
    "long": "BIGINT",
    "double": "DOUBLE",
    "int": "INTEGER",
    "timestamp": "TIMESTAMP",
}

# Named example queries (documented in the docs, used by the dashboard). Each returns the
# columns named in its ``AS`` clauses, in this order, and runs on an empty warehouse.
EXAMPLE_QUERIES: dict[str, str] = {
    # (drive_id, n_hard_brakes): every drive, zero when it has no HARD_BRAKE.
    "hard_brakes_per_drive": (
        "SELECT d.drive_id AS drive_id, COUNT(e.drive_id) AS n_hard_brakes "
        "FROM drives d LEFT JOIN events e "
        f"ON e.drive_id = d.drive_id AND e.event_type = '{EventType.HARD_BRAKE.value}' "
        "GROUP BY d.drive_id ORDER BY d.drive_id"
    ),
    # (event_type, n_events): all drives.
    "events_by_type": (
        "SELECT event_type, COUNT(*) AS n_events FROM events "
        "GROUP BY event_type ORDER BY event_type"
    ),
    # (error_code, n_frames): frames with a non-null code only (rejects is a separate table).
    "error_counts": (
        "SELECT error_code, COUNT(*) AS n_frames FROM frames "
        "WHERE error_code IS NOT NULL GROUP BY error_code ORDER BY error_code"
    ),
    # Exactly the DRIVES_COLUMNS, one row per drive.
    "drives_overview": (
        "SELECT drive_id, vehicle_id, vehicle_kind, drive_start_utc, start_ts, duration_s, "
        "n_frames, distance_m, n_events, n_errors FROM drives ORDER BY drive_id"
    ),
    # (drive_id, vehicle_id, max_speed_ms): the five drives with the highest top speed.
    "fastest_drives": (
        "SELECT drive_id, MAX(vehicle_id) AS vehicle_id, MAX(speed) AS max_speed_ms "
        "FROM frames GROUP BY drive_id ORDER BY max_speed_ms DESC, drive_id LIMIT 5"
    ),
}


def _columns(table: str) -> list[tuple[str, str]]:
    """Parse the DDL of a table into (column, DuckDB type) pairs, in DDL order.

    Args:
        table: One of `TABLES`.

    Returns:
        The column names with their DuckDB types.
    """
    pairs = (column.split() for column in TABLE_DDLS[table].split(", "))
    return [(name, DUCKDB_TYPES[spark_type]) for name, spark_type in pairs]


def _quote(text: str) -> str:
    """Quote a string as a SQL literal (a view definition cannot take bound parameters).

    Args:
        text: The raw text, for example a path.

    Returns:
        The text in single quotes with inner quotes doubled.
    """
    return "'" + text.replace("'", "''") + "'"


def _source(pattern: str, partitioned: bool) -> str:
    """Build the ``read_parquet`` call over the files of a table.

    Args:
        pattern: The glob of the Parquet files.
        partitioned: Whether the table directory holds ``drive_id=...`` partitions.

    Returns:
        The table function call with the hive options.
    """
    types = ", hive_types={'drive_id': VARCHAR}" if partitioned else ""
    return f"read_parquet({_quote(pattern)}, hive_partitioning=true{types})"


def _parquet_view(table: str, pattern: str, partitioned: bool) -> str:
    """Build the statement of a view over the Parquet files of a table.

    When the table has ``drive_id=...`` partition directories, ``drive_id`` is forced to
    VARCHAR (otherwise DuckDB infers a number for ids such as ``12``); DuckDB rejects that
    option for a table without such a partition column. The columns are selected in DDL order
    and cast to the DDL types, so every view has the shape of its DDL.

    Args:
        table: One of `TABLES`.
        pattern: The glob of the table's committed Parquet files.
        partitioned: Whether the table directory holds ``drive_id=...`` partitions.

    Returns:
        The ``CREATE OR REPLACE VIEW`` statement.
    """
    select = ", ".join(f"CAST({name} AS {kind}) AS {name}" for name, kind in _columns(table))
    return f"CREATE OR REPLACE VIEW {table} AS SELECT {select} FROM {_source(pattern, partitioned)}"


def _check_columns(con: duckdb.DuckDBPyConnection, table: str, source: str) -> None:
    """Check that the Parquet files hold every column of the table's DDL.

    Args:
        con: The connection.
        table: One of `TABLES`.
        source: The ``read_parquet`` call over the table's files.

    Raises:
        ValueError: If a DDL column is absent; the one-line message names the table and the
            columns.
    """
    described = con.execute(f"DESCRIBE SELECT * FROM {source}").df()
    found = set(described["column_name"])
    missing = [name for name, _ in _columns(table) if name not in found]
    if missing:
        raise ValueError(f"table {table}: the Parquet files have no column {', '.join(missing)}")


def _empty_view(table: str) -> str:
    """Build the statement of an empty view with the columns and types of a table's DDL.

    Args:
        table: One of `TABLES`.

    Returns:
        The ``CREATE OR REPLACE VIEW`` statement.
    """
    select = ", ".join(f"CAST(NULL AS {kind}) AS {name}" for name, kind in _columns(table))
    return f"CREATE OR REPLACE VIEW {table} AS SELECT {select} WHERE false"


class Warehouse:
    """In-process DuckDB connection with one view per warehouse table.

    The views ``frames``, ``events``, ``drives``, ``rejects`` and ``raw_frames`` read the
    committed Parquet files of each table (``<table>/drive_id=*/*.parquet`` for a partitioned
    table, ``<table>/*.parquet`` otherwise, so files a crashed Spark write left under
    ``_temporary/`` are not read) with hive partitioning; the partition column ``drive_id``
    is a string even for numeric-looking ids. A table that does not exist becomes an empty
    view with the columns and types of its DDL in ``vl.schemas``, so consumers work on an
    empty warehouse.
    """

    def __init__(self, path: str) -> None:
        """Connect in memory (no database file) and register the views.

        The path is glob-escaped (``[``, ``*`` and ``?`` in a directory name stay literal). A
        Parquet table that lacks a column of its DDL is refused with a one-line error.
        A table without any Parquet file (missing directory, empty table, or only partition
        directories with no data) is registered as an empty typed view.

        Args:
            path: Warehouse directory that holds the table directories.

        Raises:
            ValueError: If the Parquet files of a table lack a column of its DDL.
        """
        self._con = duckdb.connect()
        for table in TABLES:
            directory = Path(path) / table
            partitioned = any(directory.glob("drive_id=*"))
            # Committed files only, at the depth Spark commits them: a crashed write leaves its
            # staged files under <table>/_temporary/, which Spark and PyArrow skip and so must we.
            layout = "drive_id=*/*.parquet" if partitioned else "*.parquet"
            if any(directory.glob(layout)):
                pattern = str(Path(glob.escape(str(directory))) / layout)
                _check_columns(self._con, table, _source(pattern, partitioned))
                self._con.execute(_parquet_view(table, pattern, partitioned))
            else:
                self._con.execute(_empty_view(table))

    def query(self, sql: str, params: Sequence[object] | None = None) -> pd.DataFrame:
        """Run a SQL query.

        Args:
            sql: The statement; values are bound through ``?`` placeholders, never
                interpolated.
            params: Values for the placeholders, in order.

        Returns:
            The result as a DataFrame. Safe to call from several threads at once.
        """
        cursor = self._con.cursor()  # a connection is not thread-safe, a cursor per call is
        try:
            return cursor.execute(sql, list(params) if params is not None else None).df()
        finally:
            cursor.close()

    def tables(self) -> list[str]:
        """List the registered views.

        Returns:
            The names ``frames``, ``events``, ``drives``, ``rejects`` and ``raw_frames``.
        """
        return list(TABLES)
