"""Feature engineering shared by training, notebook and the Streamlit app.

Design rules
------------
* Only columns that would be known when a collision is *reported*
  (location, road layout, weather, lighting-equivalent fields, time) are
  used as predictors. Post-outcome columns live in ``LEAKAGE_COLUMNS``.
* All derived features are pure functions of the raw row, so the same code
  runs identically at training time and at inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import (
    CELL_SIZE_DEG,
    COUNT_TEST_START_MONTH,
    LEAKAGE_COLUMNS,
    SEVERITY_TRAIN_END,
    SEVERITY_VAL_END,
)

# Predictors for the severity model (leakage columns already excluded).
NUMERIC_FEATURES = [
    "latitude", "longitude", "temperature", "hour", "month", "lanes",
]
CATEGORICAL_FEATURES = [
    "city", "state", "road_type", "weather", "visibility", "traffic_density",
    "day_of_week", "season", "is_weekend", "is_peak_hour", "traffic_signal",
    "festival",
]
TARGET_COLUMN = "accident_severity"

SEASON_BOUNDS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "monsoon": (9, 10, 11),
}


def month_to_season(month: int) -> str:
    """Map a calendar month (1-12) to an Indian meteorological season name."""
    for name, months in SEASON_BOUNDS.items():
        if month in months:
            return name
    raise ValueError(f"Month must be 1-12, got {month}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived time and spatial features; keep only model-safe columns.

    Returns a copy with:
    ``dt`` (datetime), ``hour``, ``month``, ``weekday_num``, ``season``,
    ``cell_id`` (0.2-degree grid id) and ``festival`` filled with ``"none"``.
    """
    out = df.copy()
    out["dt"] = pd.to_datetime(out["date"], errors="coerce")
    if out["dt"].isna().any():
        bad = int(out["dt"].isna().sum())
        raise ValueError(f"{bad} rows have unparseable dates; clean the source file.")

    # hour is present in the source; re-derive from `time` only to validate.
    parsed_hour = pd.to_datetime(out["time"], format="%H:%M", errors="coerce").dt.hour
    if parsed_hour.isna().any():
        raise ValueError("Some `time` values could not be parsed as HH:MM.")
    if not (parsed_hour == out["hour"]).all():
        mismatch = int((parsed_hour != out["hour"]).sum())
        raise ValueError(f"`hour` disagrees with `time` in {mismatch} rows.")

    out["month"] = out["dt"].dt.month.astype(int)
    out["weekday_num"] = out["dt"].dt.day_of_week.astype(int)
    out["season"] = out["month"].map(month_to_season)
    # Festival is 99% missing in the raw file; "none" is an observed state
    # (no festival recorded), not an imputed measurement.
    out["festival"] = out["festival"].fillna("none").astype(str)
    out["cell_id"] = assign_grid_cells(out["latitude"], out["longitude"], CELL_SIZE_DEG)
    return out


def assign_grid_cells(lat: pd.Series, lon: pd.Series, size_deg: float = CELL_SIZE_DEG) -> pd.Series:
    """Assign each coordinate to a fixed-size latitude/longitude grid cell.

    The cell id is the SW corner rounded to the grid size, e.g. ``18.6_73.8``.
    Coordinates are validated to lie within India's bounding box first.
    """
    lat = pd.to_numeric(lat, errors="coerce")
    lon = pd.to_numeric(lon, errors="coerce")
    if lat.isna().any() or lon.isna().any():
        raise ValueError("Missing latitude/longitude values; refusing to invent coordinates.")
    if not lat.between(6.0, 38.0).all() or not lon.between(68.0, 98.0).all():
        raise ValueError("Coordinates outside the India bounding box (6-38N, 68-98E).")
    cell_lat = np.floor(lat / size_deg) * size_deg
    cell_lon = np.floor(lon / size_deg) * size_deg
    return cell_lat.round(3).astype(str) + "_" + cell_lon.round(3).astype(str)


def severity_split_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Chronological train/validation/test masks for the severity model.

    Train: <= 2023-12-31 | Validation: 2024 | Test: 2025 (partial year).
    The test period is never touched during model selection.
    """
    dt = pd.to_datetime(df["date"])
    return {
        "train": dt <= pd.Timestamp(SEVERITY_TRAIN_END),
        "val": (dt > pd.Timestamp(SEVERITY_TRAIN_END)) & (dt <= pd.Timestamp(SEVERITY_VAL_END)),
        "test": dt > pd.Timestamp(SEVERITY_VAL_END),
    }


def model_matrix_columns() -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) predictor column lists for sklearn pipelines."""
    return list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES)


def assert_no_leakage(columns: list[str]) -> None:
    """Raise if any forbidden (post-outcome) column sneaks into the predictors."""
    bad = [c for c in columns if c in LEAKAGE_COLUMNS]
    if bad:
        raise ValueError(f"Leakage columns in predictors: {bad}")


def is_period_start(period: str) -> bool:
    """True when ``period`` (YYYY-MM) is the first month of count-model testing."""
    return period == COUNT_TEST_START_MONTH
