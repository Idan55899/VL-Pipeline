"""Pytest session setup: pin Spark's Python to the interpreter running the tests."""

import os
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

# Intentionally overrides any user-set PYSPARK_PYTHON / PYSPARK_DRIVER_PYTHON.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


@pytest.fixture(scope="session")
def spark() -> "Iterator[SparkSession]":
    """Provide the one Spark session shared by the whole test run, built by ``get_spark``."""
    from vl.spark import get_spark  # lazy: collection must work before Spark is touched

    session = get_spark("vl-tests")
    yield session
    session.stop()


# Shared fixtures of the transform and warehouse tests (the golden warehouse is built once).
pytest_plugins = ["transform_helpers"]
