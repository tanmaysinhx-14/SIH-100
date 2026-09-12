"""Unit tests for the synthetic RF simulator module."""

from __future__ import annotations

import numpy as np
import pytest

from simulator import (
    SyntheticChannelSource,
    generate_awgn,
    generate_civilian_signal,
    generate_hostile_jammer,
    generate_hostile_radar,
    generate_spectrum_batch,
)


def test_generate_awgn_shape_and_dtype() -> None:
    samples = generate_awgn(num_samples=1024, noise_power_db=-20.0)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1024,)
    assert samples.dtype == np.complex128
    assert np.all(np.isfinite(samples))


def test_generate_awgn_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        generate_awgn(num_samples=0)
    with pytest.raises(ValueError):
        generate_awgn(num_samples=-10)


def test_generate_civilian_signal_shape() -> None:
    samples = generate_civilian_signal(num_samples=1024, sample_rate=1.0e6)
    assert samples.shape == (1024,)
    assert samples.dtype == np.complex128
    assert np.all(np.isfinite(samples))


def test_generate_hostile_radar_shape() -> None:
    samples = generate_hostile_radar(num_samples=1024, sample_rate=1.0e6)
    assert samples.shape == (1024,)
    assert samples.dtype == np.complex128
    assert np.all(np.isfinite(samples))


def test_generate_hostile_jammer_power() -> None:
    noise = generate_awgn(num_samples=1024, noise_power_db=-20.0)
    jammer = generate_hostile_jammer(num_samples=1024, jammer_power_db=10.0)

    noise_power = np.mean(np.abs(noise) ** 2)
    jammer_power = np.mean(np.abs(jammer) ** 2)
    assert jammer_power > noise_power * 100.0


def test_generate_spectrum_batch_default() -> None:
    batch = generate_spectrum_batch(num_channels=10)
    assert len(batch) == 10
    classes = {channel["signal_class"] for channel in batch.values()}
    # Default 10 channel batch must contain all 4 classes
    assert len(classes) == 4
    assert "Background Noise" in classes
    assert "Civilian Broadcast" in classes
    assert "Hostile Radar" in classes
    assert "Hostile Jammer" in classes


def test_synthetic_channel_source_adapter() -> None:
    source = SyntheticChannelSource(num_channels=10, seed=42)
    batch = source.read_batch()
    assert len(batch) == 10
    for channel in batch.values():
        assert "iq" in channel
        assert "sample_rate" in channel
        assert "freq_mhz" in channel
