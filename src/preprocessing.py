"""Leakage-safe preprocessing: sklearn Pipelines and ColumnTransformers.

Every transformation is fit only on training data because it is *inside* the
pipeline that scikit-learn fits with ``pipeline.fit(X_train, y_train)``.
Unseen categories at inference time are handled by
``OneHotEncoder(handle_unknown="ignore")``.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import RANDOM_SEED
from .features import assert_no_leakage, model_matrix_columns


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median-impute + standard-scale. Categorical: one-hot encode."""
    num_cols, cat_cols = model_matrix_columns()
    assert_no_leakage(num_cols + cat_cols)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # sparse_output=False keeps downstream models simple and inspectable.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        verbose_feature_names_out=False,
    )


def get_random_state() -> int:
    """Single documented random seed used by every stochastic component."""
    return RANDOM_SEED
