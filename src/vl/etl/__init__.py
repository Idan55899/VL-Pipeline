"""Pure ETL logic of the pipeline: feature derivation, event detection and drive summaries.

Role in the pipeline: the transform stage between ``raw_frames`` and the ``frames``,
``events`` and ``drives`` tables. Everything here is pandas in, pandas out, so the same
code runs inside Spark ``applyInPandas`` and in the unit tests.
"""
