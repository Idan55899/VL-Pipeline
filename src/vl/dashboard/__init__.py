"""Dash dashboard over the Parquet warehouse.

Role in the pipeline: the presentation layer. ``figures`` turns query results into Plotly
figures and ``app`` wires them to a Dash app that only reads through ``vl.warehouse``.
"""
