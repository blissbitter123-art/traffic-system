"""Collision-count forecasting per grid cell per month.

Task: given a cell and a future month, forecast the number of *reported*
collisions in that cell-month. This is a count-intensity forecast, NOT a
probability that any individual road user will crash - there is no traffic
exposure (vehicles-km) denominator in this dataset.

Baselines compared against a learned model:
  * prev_month   - carry forward last month's count.
  * seasonal_avg - mean count for the same calendar month in earlier years.
  * poisson      - scikit-learn PoissonRegressor on time-derived features.

Evaluation: MAE/RMSE on the six complete held-out months 2024-10..2025-03.

Run as a script:  python -m src.train_hotspots
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline

from . import (
    COUNT_TEST_START_MONTH,
    COUNT_TRAIN_END_MONTH,
    HOTSPOT_PIPELINE_PATH,
    OUTPUTS_DIR,
    RANDOM_SEED,
    ensure_dirs,
)
from .data_loader import load_accidents
from .evaluation import comparison_table, count_metrics
from .hotspot_analysis import (
    add_intensity_labels,
    build_monthly_panel,
    cell_coverage,
    cell_trends,
    intensity_thresholds,
    prepare_collisions,
)

MODEL_FEATURES_NUM = [
    "month_num",          # linear time trend (Period ordinal)
    "cell_month_mean",    # cell's own historical average (train window)
    "prev_month_count",   # lag-1
    "avg_same_month_prior",  # seasonal lag average (train window only)
    "cell_total_train",   # cell size proxy
]
# month_of_year (1-12) is handled separately as a one-hot categorical below.


def build_forecast_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Lag and seasonal features using ONLY information available before the month.

    ``prev_month_count`` and ``avg_same_month_prior`` are computed with a
    rolling shift inside each cell; for training rows they see earlier months
    only, which keeps the design leakage-safe by construction.
    """
    out = panel.sort_values(["cell_id", "month_num"]).copy()
    out["month_of_year"] = pd.PeriodIndex(out["month"], freq="M").month.astype(int)

    out["prev_month_count"] = (
        out.groupby("cell_id")["collision_count"].shift(1).fillna(0.0)
    )

    # Seasonal average: mean of the same calendar month in *previous* months
    # of the same cell (shifted so the current row is excluded).
    def _seasonal_avg(sub: pd.DataFrame) -> pd.Series:
        shifted = sub["collision_count"].shift(1)
        moy = sub["month_of_year"]
        return shifted.groupby(moy).transform(lambda s: s.expanding().mean())

    out["avg_same_month_prior"] = (
        out.groupby("cell_id", group_keys=False).apply(_seasonal_avg, include_groups=False)
        .fillna(out.groupby("cell_id")["collision_count"].transform("mean"))
        .reset_index(level=0, drop=True)
        .reindex(out.index)
    )

    train_mask = out["month"] <= COUNT_TRAIN_END_MONTH
    cell_train_mean = (
        out[train_mask].groupby("cell_id")["collision_count"].mean().rename("cell_month_mean")
    )
    cell_train_total = (
        out[train_mask].groupby("cell_id")["collision_count"].sum().rename("cell_total_train")
    )
    out["cell_month_mean"] = out["cell_id"].map(cell_train_mean).fillna(0.0)
    out["cell_total_train"] = out["cell_id"].map(cell_train_total).fillna(0.0)
    return out


def train_and_evaluate_counts() -> dict[str, Any]:
    """Fit the Poisson count model, score vs baselines, export artefacts."""
    ensure_dirs()
    collisions = prepare_collisions(load_accidents())
    panel, cell_stats = build_monthly_panel(collisions)
    cuts = intensity_thresholds(
        panel.loc[panel["month"] <= COUNT_TRAIN_END_MONTH, "collision_count"].values
    )
    panel = add_intensity_labels(panel, cuts)
    feats = build_forecast_features(panel)

    train = feats[feats["month"] <= COUNT_TRAIN_END_MONTH]
    test = feats[
        (feats["month"] >= COUNT_TEST_START_MONTH)
        & (feats["is_test_period"])
    ].copy()

    # ---- learned model ------------------------------------------------------
    pre = ColumnTransformer([
        ("num", StandardScaler(), MODEL_FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["month_of_year"]),
    ])
    model = SKPipeline([
        ("prep", pre),
        ("poisson", PoissonRegressor(alpha=1e-4, max_iter=500)),  # deterministic: no random_state param
    ])
    model.fit(train[MODEL_FEATURES_NUM + ["month_of_year"]], train["collision_count"])

    test["pred_poisson"] = np.clip(
        model.predict(test[MODEL_FEATURES_NUM + ["month_of_year"]]), 0, None
    )
    # ---- baselines ----------------------------------------------------------
    test["pred_prev_month"] = test["prev_month_count"].clip(lower=0)
    test["pred_seasonal_avg"] = test["avg_same_month_prior"].fillna(
        test["cell_month_mean"]
    ).clip(lower=0)
    # persistence of the cell's historical mean as a third simple baseline
    test["pred_cell_mean"] = test["cell_month_mean"]

    y_true = test["collision_count"].values
    results = {
        "poisson": count_metrics(y_true, test["pred_poisson"]),
        "prev_month": count_metrics(y_true, test["pred_prev_month"]),
        "seasonal_avg": count_metrics(y_true, test["pred_seasonal_avg"]),
        "cell_mean": count_metrics(y_true, test["pred_cell_mean"]),
    }
    comparison = comparison_table(results)
    best_name = min(results, key=lambda n: results[n]["mae"])  # lowest MAE wins

    # ---- artefacts ----------------------------------------------------------
    joblib.dump(
        {"model": model, "features": MODEL_FEATURES_NUM, "cuts": cuts},
        HOTSPOT_PIPELINE_PATH,
    )
    comparison.reset_index(names="model").to_csv(
        OUTPUTS_DIR / "count_model_comparison.csv", index=False
    )

    coverage = cell_coverage(panel)
    trends = cell_trends(panel)

    # hotspot_cells.csv = one row per eligible cell with history + status
    hist = (
        panel.groupby("cell_id")
        .agg(
            total_collisions=("collision_count", "sum"),
            mean_monthly=("collision_count", "mean"),
            last_3m=("collision_count", lambda s: s.sort_values().tail(3).sum()),
            first_month=("month", "min"),
            last_month=("month", "max"),
        )
        .reset_index()
    )
    hotspot_cells = (
        hist.merge(coverage.drop(columns=["total_collisions"]), on="cell_id")
        .merge(trends, on="cell_id", how="left")
    )
    lat_lon = collisions.groupby("cell_id")[["latitude", "longitude"]].mean().reset_index()
    hotspot_cells = hotspot_cells.merge(lat_lon, on="cell_id", how="left")
    hotspot_cells.to_csv(OUTPUTS_DIR / "hotspot_cells.csv", index=False)

    payload = {
        "best_model": best_name,
        "metrics": results,
        "test_period": f"{COUNT_TEST_START_MONTH} to 2025-03 (6 complete months)",
        "train_period": f"2022-01 to {COUNT_TRAIN_END_MONTH}",
        "n_cell_months_train": int(len(train)),
        "n_cell_months_test": int(len(test)),
        "n_eligible_cells": int(panel["cell_id"].nunique()),
        "intensity_cuts": cuts,
        "random_seed": RANDOM_SEED,
    }
    (OUTPUTS_DIR / "count_metrics.json").write_text(json.dumps(payload, indent=2))

    return {
        "payload": payload,
        "comparison": comparison,
        "panel": panel,
        "features": feats,
        "test": test,
        "hotspot_cells": hotspot_cells,
        "model": model,
    }


if __name__ == "__main__":
    out = train_and_evaluate_counts()
    print("Best count model:", out["payload"]["best_model"])
    print(out["comparison"].round(3))
    print("\nTrend labels:")
    print(out["hotspot_cells"]["trend_label"].value_counts())
