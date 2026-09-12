"""Unit tests for signal feature extraction and Random Forest classifier."""

from __future__ import annotations

import numpy as np
import pytest

from classifier import RFSignalClassifier, extract_signal_features
from simulator import generate_awgn, generate_hostile_radar


def test_extract_signal_features_shape_and_bounds() -> None:
    iq = generate_awgn(num_samples=1024)
    features = extract_signal_features(iq, sample_rate=1.0e6)

    assert isinstance(features, np.ndarray)
    assert features.shape == (9,)
    assert np.all(np.isfinite(features))


def test_classifier_train_predict() -> None:
    classifier = RFSignalClassifier(random_state=42)
    classifier.train(samples_per_class=20)
    assert classifier.is_trained

    test_signal = generate_hostile_radar(num_samples=1024)
    result = classifier.predict(test_signal, sample_rate=1.0e6)

    assert "classification" in result
    assert "confidence" in result
    assert "threat_level" in result
    assert "status" in result
    assert "features" in result
    assert result["threat_level"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def test_classifier_evaluate_model() -> None:
    classifier = RFSignalClassifier(random_state=42)
    classifier.train(samples_per_class=20)

    eval_result = classifier.evaluate_model(samples_per_class=10, seed=123)
    assert "accuracy" in eval_result
    assert "classification_report" in eval_result
    assert "confusion_matrix" in eval_result
    assert "feature_importances" in eval_result
    assert 0.0 <= eval_result["accuracy"] <= 1.0
