"""Build models/metadata.json - the single manifest the Streamlit app loads.

Run after both training scripts:
    python -m src.train_severity
    python -m src.train_hotspots
    python -m src.build_metadata
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import (
    DATASET_GEOGRAPHY,
    DATASET_NAME,
    DATASET_PERIOD,
    METADATA_PATH,
    MODELS_DIR,
    N_RECORDS,
    OUTPUTS_DIR,
    SEVERITY_CLASSES,
    SEVERITY_WEIGHTS,
    ensure_dirs,
)
from .data_loader import data_source_register


def build_metadata() -> dict:
    """Merge severity + count results and the data register into one JSON."""
    ensure_dirs()

    severity_metrics = json.loads((OUTPUTS_DIR / "metrics.json").read_text())
    count_metrics = json.loads((OUTPUTS_DIR / "count_metrics.json").read_text())

    metadata = {
        "project": "Traffic Safety Intelligence - Accident Risk Prediction & Hotspot Analysis",
        "built_on": str(date.today()),
        "dataset": {
            "name": DATASET_NAME,
            "period": DATASET_PERIOD,
            "geography": DATASET_GEOGRAPHY,
            "records": N_RECORDS,
            "source_register": data_source_register(),
        },
        "severity_model": severity_metrics,
        "count_model": count_metrics,
        "severity_classes": SEVERITY_CLASSES,
        "severity_weights": SEVERITY_WEIGHTS,
        "artifacts": {
            "severity_pipeline": "models/severity_pipeline.joblib",
            "hotspot_count_pipeline": "models/hotspot_count_pipeline.joblib",
            "metrics": "outputs/metrics.json",
            "classification_report": "outputs/classification_report.csv",
            "model_comparison": "outputs/model_comparison.csv",
            "hotspot_cells": "outputs/hotspot_cells.csv",
            "count_model_comparison": "outputs/count_model_comparison.csv",
        },
        "usage_caveat": (
            "Severity predictions are conditional on a reported collision; they "
            "are NOT proof a collision will occur at a location. Count forecasts "
            "are reported-collision intensity, not per-vehicle accident "
            "probability (no traffic-exposure denominator available)."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata written to {METADATA_PATH}")
    return metadata


if __name__ == "__main__":
    build_metadata()
