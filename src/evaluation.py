"""Evaluation helpers: severity classification and count forecasting metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)


def classification_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Macro-F1, balanced accuracy, plain accuracy, log-loss placeholder.

    Accuracy is reported for completeness but is explicitly *not* the selection
    metric: with ~55% minor / 30% major / 15% fatal, a majority-class dummy
    already reaches ~55% accuracy while being useless for the rare fatal class.
    """
    y_true = np.asarray(y_true)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def full_classification_report(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-class precision/recall/F1/support as a DataFrame."""
    return pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    ).T


def confusion_df(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Confusion matrix with readable row/column labels."""
    return pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )


def probability_metrics(y_true: pd.Series | np.ndarray, y_proba: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Log-loss and Brier score for predicted class probabilities."""
    y_true = np.asarray(y_true)
    class_to_idx = {c: i for i, c in enumerate(labels)}
    y_idx = np.array([class_to_idx[v] for v in y_true])
    # Multi-class Brier: mean squared error between one-hot truth and probabilities.
    one_hot = np.eye(len(labels))[y_idx]
    brier = float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))
    return {
        "log_loss": float(log_loss(y_idx, y_proba, labels=list(range(len(labels))))),
        "brier_multiclass": brier,
    }


def count_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / MAPE-style metrics for monthly collision-count forecasts."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    # Pinball-style directional check: does the model at least track the mean?
    return {"mae": mae, "rmse": rmse, "mape_pct": round(mape, 2) if mape == mape else float("nan")}


def comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn {model_name: {metric: value}} into a sorted comparison DataFrame."""
    table = pd.DataFrame(results).T
    return table.sort_values(by=list(table.columns)[0], ascending=False)
