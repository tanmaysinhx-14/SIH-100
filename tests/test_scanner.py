"""Unit tests for Welch PSD measurement, scoring, and scanner module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanner import (
    THREAT_MULTIPLIERS,
    calculate_psd_metrics,
    estimate_noise_floor,
    get_priority_breakdown,
    scan_and_prioritize,
)
from simulator import generate_awgn, generate_spectrum_batch


def test_calculate_psd_metrics() -> None:
    iq = generate_awgn(num_samples=1024, noise_power_db=-10.0)
    metrics = calculate_psd_metrics(iq, sample_rate=1.0e6)

    assert "freqs" in metrics
    assert "psd" in metrics
    assert "psd_db" in metrics
    assert "mean_power_db" in metrics
    assert metrics["nperseg"] == 256
    assert len(metrics["freqs"]) == 256
    assert len(metrics["psd"]) == 256
    assert np.all(np.isfinite(metrics["psd"]))


def test_scan_and_prioritize_empty_batch() -> None:
    df = scan_and_prioritize({}, threshold_db=-10.0)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_scan_and_prioritize_ranking() -> None:
    batch = generate_spectrum_batch(num_channels=10, rng=np.random.default_rng(42))
    df = scan_and_prioritize(batch, threshold_db=-10.0)

    assert not df.empty
    assert len(df) == 10
    # Assert monotonic descending order of Priority Score
    scores = df["Priority Score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_get_priority_breakdown() -> None:
    breakdown = get_priority_breakdown(
        power_db=5.0,
        threshold_db=-10.0,
        threat_level="CRITICAL",
        classification="Hostile Jammer",
    )
    assert breakdown["power_excess_db"] == 15.0
    assert breakdown["threat_multiplier"] == 2.5
    assert breakdown["priority_score"] == 37.5
    assert "explanation" in breakdown


def test_estimate_noise_floor() -> None:
    batch = generate_spectrum_batch(num_channels=10, rng=np.random.default_rng(42))
    threshold = estimate_noise_floor(batch)
    assert isinstance(threshold, float)
    assert -25.0 <= threshold <= 10.0
