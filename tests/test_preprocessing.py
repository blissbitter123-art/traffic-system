"""Tests for feature engineering and preprocessing."""

import pandas as pd
import pytest

from src.features import (
    assign_grid_cells,
    engineer_features,
    month_to_season,
    severity_split_masks,
)
from src.preprocessing import build_preprocessor


def _sample_df(n: int = 50) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "accident_id": i,
            "city": ["Mumbai", "Delhi", "Pune"][i % 3],
            "state": ["Maharashtra", "Delhi", "Maharashtra"][i % 3],
            "latitude": 19.0 + (i % 5) * 0.01,
            "longitude": 72.8 + (i % 5) * 0.01,
            "date": "2023-06-15",
            "time": "10:00",
            "hour": 10,
            "day_of_week": "Thursday",
            "is_weekend": 0,
            "road_type": "urban",
            "lanes": 2,
            "traffic_signal": 1,
            "weather": "clear",
            "visibility": "high",
            "temperature": 30,
            "traffic_density": "medium",
            "cause": "distraction",
            "accident_severity": ["minor", "major", "fatal"][i % 3],
            "vehicles_involved": 2,
            "casualties": 1,
            "is_peak_hour": 0,
            "festival": None,
            "risk_score": 0.4,
        })
    return pd.DataFrame(rows)


def test_month_to_season():
    assert month_to_season(1) == "winter"
    assert month_to_season(5) == "spring"
    assert month_to_season(7) == "summer"
    assert month_to_season(10) == "monsoon"
    with pytest.raises(ValueError):
        month_to_season(13)


def test_engineer_features_adds_columns_and_fills_festival():
    out = engineer_features(_sample_df())
    for col in ["dt", "month", "season", "cell_id", "weekday_num"]:
        assert col in out.columns
    # Festival missing -> "none" (observed state, not an invented measurement).
    assert (out["festival"] == "none").all()


def test_engineer_features_rejects_bad_coordinates():
    df = _sample_df()
    df.loc[0, "latitude"] = 999.0
    with pytest.raises(ValueError):
        engineer_features(df)


def test_grid_cells_are_stable():
    lat = pd.Series([19.05, 19.15, 12.95])
    lon = pd.Series([72.85, 72.95, 80.25])
    cells = assign_grid_cells(lat, lon, 0.2)
    assert cells.iloc[0] == cells.iloc[1]  # 19.05 & 19.15 both fall in [19.0,19.2)
    assert cells.iloc[0] != cells.iloc[2]


def test_severity_split_is_chronological_and_disjoint():
    df = engineer_features(_sample_df(90))
    # 30 rows in each split window: train 2023, val 2024, test 2025
    df["date"] = ["2023-01-01", "2024-06-15", "2025-02-15"] * 30
    df = engineer_features(df)
    masks = severity_split_masks(df)
    assert not (masks["train"] & masks["val"]).any()
    assert not (masks["train"] & masks["test"]).any()
    assert not (masks["val"] & masks["test"]).any()
    assert masks["train"].sum() == 30
    assert masks["val"].sum() == 30
    assert masks["test"].sum() == 30


def test_preprocessor_handles_unknown_categories():
    pre = build_preprocessor()
    train = _sample_df(30)
    out = engineer_features(train)
    num_cols, cat_cols = ["latitude", "longitude", "temperature", "hour", "month", "lanes"], [
        "city", "state", "road_type", "weather", "visibility", "traffic_density",
        "day_of_week", "season", "is_weekend", "is_peak_hour", "traffic_signal", "festival",
    ]
    pre.fit(out[num_cols + cat_cols])
    # unseen city value must not raise
    row = out[num_cols + cat_cols].copy()
    row.loc[row.index[0], "city"] = "Atlantis"
    transformed = pre.transform(row)
    assert transformed.shape[0] == len(row)
