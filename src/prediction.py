"""Inference helpers used by the Streamlit app and the tests.

All functions load *saved* artefacts; nothing is retrained here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import (
    METADATA_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    SEVERITY_CLASSES,
    SEVERITY_PIPELINE_PATH,
    SEVERITY_WEIGHTS,
    HOTSPOT_PIPELINE_PATH,
)
from .features import model_matrix_columns


@lru_cache(maxsize=None)
def load_severity_model() -> Any:
    """Load the saved severity pipeline (preprocessing + classifier)."""
    if not SEVERITY_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Severity model not found at {SEVERITY_PIPELINE_PATH}. "
            "Run `python -m src.train_severity` first."
        )
    return joblib.load(SEVERITY_PIPELINE_PATH)


@lru_cache(maxsize=None)
def load_count_artifacts() -> dict[str, Any]:
    """Load the saved count-forecast artefacts (model, feature list, cuts)."""
    if not HOTSPOT_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Count model not found at {HOTSPOT_PIPELINE_PATH}. "
            "Run `python -m src.train_hotspots` first."
        )
    return joblib.load(HOTSPOT_PIPELINE_PATH)


def load_metrics() -> dict[str, Any]:
    """Load saved evaluation metrics written by the training scripts."""
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text())


def predict_severity(input_record: dict[str, Any]) -> dict[str, Any]:
    """Predict severity class probabilities for ONE accident-condition record.

    Parameters
    ----------
    input_record:
        Dict with the raw feature columns (city, weather, hour, ...).

    Returns
    -------
    dict with 'classes', 'probabilities' and 'predicted_class'.
    Unknown categories are handled by the OneHotEncoder's ignore setting.
    """
    pipe = load_severity_model()
    num_cols, cat_cols = model_matrix_columns()
    row = pd.DataFrame([input_record])[num_cols + cat_cols]
    proba = pipe.predict_proba(row)[0]
    classes = list(pipe.named_steps["clf"].classes_)
    # Re-order to the canonical SEVERITY_CLASSES order if it differs.
    ordered = [float(proba[list(classes).index(c)]) for c in SEVERITY_CLASSES]
    return {
        "classes": SEVERITY_CLASSES,
        "probabilities": ordered,
        "predicted_class": SEVERITY_CLASSES[int(np.argmax(ordered))],
    }


def expected_harm_index(
    forecast_count: float,
    severity_probabilities: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Transparent expected-harm index (a prioritisation score, NOT a probability).

    index = forecast_count x sum_c P(c) * weight_c

    Weights default to minor=1, major=4, fatal=10 and are a documented policy
    choice, editable by the user in the app.
    """
    weights = weights or SEVERITY_WEIGHTS
    weighted = sum(severity_probabilities.get(c, 0.0) * w for c, w in weights.items())
    return float(forecast_count) * float(weighted)


def forecast_cell_month(cell_id: str, month: str, panel: pd.DataFrame) -> float:
    """Forecast collision count for one cell-month using the saved model.

    ``panel`` must contain the historical cell-month counts (from
    ``hotspot_analysis.build_monthly_panel``) so lag features can be rebuilt
    exactly as during training.
    """
    from .train_hotspots import MODEL_FEATURES_NUM, build_forecast_features

    arts = load_count_artifacts()
    model = arts["model"]

    cell_hist = panel[panel["cell_id"] == cell_id].sort_values("month_num")
    if cell_hist.empty:
        raise ValueError(f"Cell '{cell_id}' is not in the documented study area.")

    period = pd.Period(month, freq="M")
    row = pd.DataFrame([{
        "cell_id": cell_id,
        "month": str(period),
        "month_num": int(period.ordinal),
        "is_test_period": True,
        "collision_count": np.nan,  # unknown future; lags come from history
    }])
    full = pd.concat([cell_hist.drop(columns=["is_test_period"], errors="ignore"), row],
                     ignore_index=True)
    feats = build_forecast_features(full)
    target = feats[feats["month"] == str(period)]
    pred = float(np.clip(model.predict(target[MODEL_FEATURES_NUM + ["month_of_year"]])[0], 0, None))
    return pred


def default_input_record() -> dict[str, Any]:
    """A reasonable default scenario record for the app's severity form."""
    return {
        "city": "Mumbai",
        "state": "Maharashtra",
        "latitude": 19.07,
        "longitude": 72.87,
        "temperature": 30,
        "hour": 9,
        "month": 6,
        "lanes": 2,
        "road_type": "urban",
        "weather": "clear",
        "visibility": "high",
        "traffic_density": "medium",
        "day_of_week": "Monday",
        "season": "summer",
        "is_weekend": 0,
        "is_peak_hour": 0,
        "traffic_signal": 1,
        "festival": "none",
    }


def model_status() -> dict[str, Any]:
    """Which artefacts exist and when they were produced (for the app header)."""
    status: dict[str, Any] = {"severity_pipeline": SEVERITY_PIPELINE_PATH.exists(),
                              "hotspot_pipeline": HOTSPOT_PIPELINE_PATH.exists(),
                              "metadata": METADATA_PATH.exists()}
    for key, path in [("severity_pipeline", SEVERITY_PIPELINE_PATH),
                      ("hotspot_pipeline", HOTSPOT_PIPELINE_PATH)]:
        if status[key]:
            status[key + "_size_kb"] = round(path.stat().st_size / 1024, 1)
    return status
