"""Model explainability: permutation importance + local probability
contributions, implemented with scikit-learn only (no heavy SHAP dependency).

Two complementary views are provided:

* ``permutation_importance_table`` - global: how much does each feature hurt
  validation performance when shuffled? (model-behaviour evidence, not
  causal proof).
* ``local_contribution`` - local: for one input row, compare the predicted
  class probability against the same prediction made with each feature
  replaced by its training-set median/mode ("what-if ablation").
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import RANDOM_SEED, SEVERITY_CLASSES
from .features import model_matrix_columns
from .preprocessing import build_preprocessor


def permutation_importance_table(
    pipeline: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Permutation importance on the VALIDATION split (never the test split).

    Works on the raw (un-preprocessed) frame because the pipeline embeds the
    preprocessor; sklearn shuffles each original column end-to-end.
    """
    result = permutation_importance(
        pipeline, X_val, y_val,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        scoring="f1_macro",
        n_jobs=-1,
    )
    num_cols, cat_cols = model_matrix_columns()
    table = pd.DataFrame({
        "feature": num_cols + cat_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return table.reset_index(drop=True)


def local_contribution(
    pipeline: Any,
    X_row: pd.DataFrame,
    background: pd.DataFrame,
    target_class: str | None = None,
) -> pd.DataFrame:
    """Ablation-based local explanation for a single predicted record.

    For each feature, replace the row's value with the background median
    (numeric) or mode (categorical), re-predict, and report the change in the
    target-class probability. Positive delta => the feature *increased* the
    model's probability for that class relative to a typical value.

    This explains model behaviour only; it is not causal evidence.
    """
    num_cols, cat_cols = model_matrix_columns()
    classes = list(pipeline.named_steps["clf"].classes_)
    if target_class is None:
        target_class = str(pipeline.predict(X_row)[0])
    idx = classes.index(target_class)

    def proba_of(frame: pd.DataFrame) -> float:
        return float(pipeline.predict_proba(frame)[idx])

    base = proba_of(X_row)
    rows = []
    for col in num_cols + cat_cols:
        modified = X_row.copy()
        if col in num_cols:
            modified[col] = float(background[col].median())
        else:
            modified[col] = background[col].mode().iloc[0]
        p = proba_of(modified)
        rows.append({
            "feature": col,
            "value": X_row[col].iloc[0],
            "baseline_value": (float(background[col].median()) if col in num_cols
                               else background[col].mode().iloc[0]),
            "proba_with_value": round(base, 4),
            "proba_if_baseline": round(p, 4),
            "contribution": round(base - p, 4),
        })
    out = pd.DataFrame(rows)
    out["target_class"] = target_class
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def fit_background_summary(X_train: pd.DataFrame) -> pd.DataFrame:
    """Tiny background frame (median/mode per column) used for ablations.

    Storing medians rather than thousands of rows keeps the saved artefacts
    small while giving every local explanation a consistent reference point.
    """
    num_cols, cat_cols = model_matrix_columns()
    summary = {}
    for col in num_cols:
        summary[col] = X_train[col].median()
    for col in cat_cols:
        summary[col] = X_train[col].mode().iloc[0]
    return pd.DataFrame([summary])


if __name__ == "__main__":
    # Regenerate outputs/permutation_importance.csv from the saved pipeline.
    from . import OUTPUTS_DIR, ensure_dirs
    from .prediction import load_severity_model
    from .train_severity import prepare_severity_data

    ensure_dirs()
    data = prepare_severity_data()
    pipe = load_severity_model()
    val_X, val_y = data["splits"]["val"]["X"], data["splits"]["val"]["y"]
    table = permutation_importance_table(pipe, val_X, val_y, n_repeats=5)
    out_path = OUTPUTS_DIR / "permutation_importance.csv"
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved to {out_path}")
