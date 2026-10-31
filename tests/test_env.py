"""Environment smoke test: Spark workers and pandas must interoperate."""

import pandas as pd
from pyspark.sql import SparkSession


def test_spark_pandas_roundtrip(spark: SparkSession) -> None:
    """A pandas DataFrame survives createDataFrame and groupBy().applyInPandas()."""

    def double_v(pdf: pd.DataFrame) -> pd.DataFrame:
        """Return the group with column ``v`` doubled (defined locally so it pickles by value)."""
        return pdf.assign(v=pdf["v"] * 2)

    source = pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    result = (
        spark.createDataFrame(source)
        .groupBy("k")
        .applyInPandas(double_v, schema="k string, v long")
        .toPandas()
        .sort_values(["k", "v"])
        .reset_index(drop=True)
    )
    expected = pd.DataFrame({"k": ["a", "a", "b"], "v": [2, 4, 6]})
    pd.testing.assert_frame_equal(result, expected)
