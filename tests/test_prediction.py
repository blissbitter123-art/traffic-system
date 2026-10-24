"""Tests for inference helpers against the saved artefacts.

These tests require the trained models to exist:
    python -m src.train_severity && python -m src.train_hotspots
"""

import numpy as np
import pytest

from src import SEVERITY_CLASSES
from src.prediction import (
    default_input_record,
    expected_harm_index,
    load_severity_model,
    predict_severity,
)


def test_severity_model_loads():
    pipe = load_severity_model()
    assert hasattr(pipe, "predict_proba")


def test_predict_severity_returns_valid_probabilities():
    result = predict_severity(default_input_record())
    assert result["classes"] == SEVERITY_CLASSES
    probs = result["probabilities"]
    assert len(probs) == 3
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-6
    assert result["predicted_class"] in SEVERITY_CLASSES


def test_predict_severity_handles_unknown_category():
    rec = default_input_record()
    rec["city"] = "Gotham"          # never seen in training
    rec["weather"] = "meteor"       # never seen in training
    result = predict_severity(rec)
    assert abs(sum(result["probabilities"]) - 1.0) < 1e-6


def test_predict_severity_handles_missing_categorical():
    rec = default_input_record()
    rec["festival"] = None
    result = predict_severity(rec)
    assert abs(sum(result["probabilities"]) - 1.0) < 1e-6


def test_expected_harm_index_math():
    # 4 collisions, 50% minor (w=1), 50% fatal (w=10) -> 4 * (0.5*1 + 0.5*10) = 22
    idx = expected_harm_index(4.0, {"minor": 0.5, "major": 0.0, "fatal": 0.5})
    assert idx == pytest.approx(22.0)


def test_expected_harm_index_respects_custom_weights():
    idx = expected_harm_index(1.0, {"minor": 1.0, "major": 0.0, "fatal": 0.0},
                              weights={"minor": 2.0, "major": 5.0, "fatal": 20.0})
    assert idx == pytest.approx(2.0)


def test_probability_ordering_is_canonical():
    # predict_severity must return probabilities in SEVERITY_CLASSES order,
    # regardless of the classifier's internal alphabetical classes_.
    rec = default_input_record()
    result = predict_severity(rec)
    assert result["classes"] == ["minor", "major", "fatal"]
    argmax_class = SEVERITY_CLASSES[int(np.argmax(result["probabilities"]))]
    assert result["predicted_class"] == argmax_class
