"""huforecast.analysis — exploratory data analysis for Deliverable 5.

This subpackage holds **read-only investigation** code that measures the
predictive power of alternative TARGETS, UNITS and FEATURES for antimicrobial
demand forecasting. It does NOT change the production pipeline
(ingestion/processing/features/modeling) and it does NOT ship the final model.

Entry point:
    python -m huforecast.analysis.features_investigation

It writes measurement tables to stdout and saves EDA figures to docs/figuras/.
All randomness is seeded (SEED = 42). No absolute paths — everything is routed
through huforecast.config.
"""
from __future__ import annotations

SEED = 42

__all__ = ["SEED"]
