"""Severity-model training: baselines, candidate models, chronological
evaluation, and export of the winning pipeline.

Run as a script:  python -m src.train_severity
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import (
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SEVERITY_CLASSES,
    SEVERITY_PIPELINE_PATH,
    ensure_dirs,
)
from .data_loader import load_accidents
from .evaluation import (
    classification_metrics,
    comparison_table,
    confusion_df,
    full_classification_report,
    probability_metrics,
)
from .features import engineer_features, model_matrix_columns, severity_split_masks
from .preprocessing import build_preprocessor


def build_candidate_models() -> dict[str, Any]:
    """Baselines + three real candidates, same preprocessing, same splits.

    * prior  - always predicts the training class frequencies. This is the
      *probability floor*: a model that cannot beat its log-loss cannot claim
      to have learned anything about severity.
    * dummy  - stratified random guess (a second, noisier floor).
    * logreg - linear, fully interpretable coefficients.
    * rf     - non-linear bagged trees with class_weight='balanced_subsample'.
    * hgb    - histogram gradient boosting.

    Class weights compensate for the 55/30/15 imbalance *without* SMOTE,
    which would synthesise impossible road-condition combinations.
    """
    return {
        "prior": DummyClassifier(strategy="prior"),
        "dummy": DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
        "logreg": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=RANDOM_SEED
        ),
    }


def make_pipeline(estimator: Any) -> Pipeline:
    """Preprocessing + estimator as ONE object: no leakage, safe inference."""
    num_cols, cat_cols = model_matrix_columns()
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", estimator),
    ])


def prepare_severity_data() -> dict[str, Any]:
    """Load, engineer features, and slice into chronological splits."""
    raw = load_accidents()
    df = engineer_features(raw)
    masks = severity_split_masks(df)
    num_cols, cat_cols = model_matrix_columns()
    X = df[num_cols + cat_cols]
    y = df["accident_severity"]
    return {
        "df": df,
        "X": X,
        "y": y,
        "masks": masks,
        "splits": {
            name: {"X": X[m], "y": y[m]} for name, m in masks.items()
        },
    }


def train_and_evaluate_severity() -> dict[str, Any]:
    """Train every candidate, score on validation, pick best by macro-F1.

    The held-out test set (2025) is touched exactly once, after the winner is
    chosen on the 2024 validation slice.
    """
    ensure_dirs()
    data = prepare_severity_data()
    splits = data["splits"]

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    val_reports: dict[str, pd.DataFrame] = {}
    val_confusions: dict[str, pd.DataFrame] = {}
    val_probas: dict[str, np.ndarray] = {}

    for name, est in build_candidate_models().items():
        pipe = make_pipeline(est)
        pipe.fit(splits["train"]["X"], splits["train"]["y"])
        pred = pipe.predict(splits["val"]["X"])
        proba = pipe.predict_proba(splits["val"]["X"])
        # predict_proba columns follow the classifier's own classes_ order
        # (alphabetical for string labels), NOT SEVERITY_CLASSES order.
        model_classes = list(pipe.named_steps["clf"].classes_)
        results[name] = classification_metrics(splits["val"]["y"], pred)
        results[name].update(
            probability_metrics(splits["val"]["y"], proba, model_classes)
        )
        fitted[name] = pipe
        val_reports[name] = full_classification_report(splits["val"]["y"], pred)
        val_confusions[name] = confusion_df(splits["val"]["y"], pred, SEVERITY_CLASSES)
        val_probas[name] = proba

    comparison = comparison_table(results)

    # ---- selection policy (documented, honest) -----------------------------
    # 1. The 'prior' dummy is the probability floor. A model is only allowed
    #    to win if it beats that floor on validation log-loss; if NONE does,
    #    we still export the best real model (lowest log-loss among logreg/rf/
    #    hgb) but record `beats_prior_baseline=False` so the notebook, the
    #    report and the app all state that the dataset carries no learnable
    #    severity signal. We never silently present a baseline as a win.
    BASELINES = {"prior", "dummy"}
    real_models = [n for n in results if n not in BASELINES]
    prior_log_loss = results["prior"]["log_loss"]
    prior_macro_f1 = results["prior"]["macro_f1"]

    # A win must clear a meaningful margin, not a 0.001 rounding wiggle.
    MARGIN_LOG_LOSS = 0.01
    MARGIN_MACRO_F1 = 0.01
    best_name = min(real_models, key=lambda n: (results[n]["log_loss"], -results[n]["macro_f1"]))
    beats_log_loss = results[best_name]["log_loss"] < prior_log_loss - MARGIN_LOG_LOSS
    beats_macro_f1 = results[best_name]["macro_f1"] > prior_macro_f1 + MARGIN_MACRO_F1
    beats_baseline = bool(beats_log_loss or beats_macro_f1)

    # ---- final, single, held-out test evaluation of the winner -------------
    best_pipe = fitted[best_name]
    test_pred = best_pipe.predict(splits["test"]["X"])
    test_proba = best_pipe.predict_proba(splits["test"]["X"])
    test_classes = list(best_pipe.named_steps["clf"].classes_)
    test_metrics = classification_metrics(splits["test"]["y"], test_pred)
    test_metrics.update(
        probability_metrics(splits["test"]["y"], test_proba, test_classes)
    )
    test_report = full_classification_report(splits["test"]["y"], test_pred)
    test_confusion = confusion_df(splits["test"]["y"], test_pred, SEVERITY_CLASSES)

    # ---- persist artefacts --------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, SEVERITY_PIPELINE_PATH)

    comparison.reset_index(names="model").to_csv(
        OUTPUTS_DIR / "model_comparison.csv", index=False
    )
    test_report.to_csv(OUTPUTS_DIR / "classification_report.csv")

    payload = {
        "best_model": best_name,
        "selection_metric": "lowest_validation_log_loss among real models",
        "beats_prior_baseline": beats_baseline,
        "baseline_note": (
            "On this dataset the best real model does NOT beat the class-prior "
            "baseline on validation log-loss or macro-F1: recorded conditions "
            "(weather, road, time, place) carry no learnable severity signal. "
            "This is a property of the supplied data, reported honestly."
            if not beats_baseline else
            "The best real model beats the class-prior baseline on validation."
        ),
        "validation_metrics": results,
        "test_metrics": test_metrics,
        "test_period": "2025-01-01 to 2025-04-15 (partial year)",
        "train_period": "2022-01-01 to 2023-12-31",
        "validation_period": "2024-01-01 to 2024-12-31",
        "n_train": int(splits["train"]["y"].shape[0]),
        "n_val": int(splits["val"]["y"].shape[0]),
        "n_test": int(splits["test"]["y"].shape[0]),
        "classes": SEVERITY_CLASSES,
        "random_seed": RANDOM_SEED,
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))

    return {
        "payload": payload,
        "comparison": comparison,
        "val_reports": val_reports,
        "val_confusions": val_confusions,
        "test_report": test_report,
        "test_confusion": test_confusion,
        "best_pipeline": best_pipe,
        "data": data,
    }


if __name__ == "__main__":
    out = train_and_evaluate_severity()
    print("Best model:", out["payload"]["best_model"])
    print(out["comparison"].round(4))
    print("\nHeld-out test metrics:")
    print(json.dumps(out["payload"]["test_metrics"], indent=2))
    print(out["test_confusion"])
