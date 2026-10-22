"""huforecast — antimicrobial demand forecasting pipeline for HU-UFPB.

Layered architecture (data engineering style):
    ingestion  -> read raw sources untouched
    processing -> clean, type, deduplicate
    features   -> build the monthly demand time series
    modeling   -> temporal split + baselines + metrics

The orchestration helper `run_pipeline` wires the layers together.
"""
from __future__ import annotations

from . import config
from .ingestion.loader import load_raw_treatments
from .processing.cleaner import clean_treatments
from .features.timeseries import monthly_demand_by_drug, total_monthly_demand
from .modeling.baseline import run_baselines

__all__ = [
    "config",
    "load_raw_treatments",
    "clean_treatments",
    "monthly_demand_by_drug",
    "total_monthly_demand",
    "run_baselines",
    "run_pipeline",
]


def run_pipeline():
    """End-to-end: raw -> clean -> series -> baselines. Returns a dict of artifacts."""
    raw = load_raw_treatments()
    clean = clean_treatments(raw)
    series = monthly_demand_by_drug(clean)
    baselines = run_baselines(series)
    return {"raw": raw, "clean": clean, "series": series, "baselines": baselines}
