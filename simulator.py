"""Synthetic RF/IQ signal generation for the DSAS prototype.

The simulator deliberately has no SDR or hardware dependencies. Every signal
is a complex NumPy array so the rest of the application can exercise the same
DSP and classification path it would use for captured IQ samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_NUM_SAMPLES = 1024
DEFAULT_SAMPLE_RATE = 1.0e6
DEFAULT_BASE_FREQUENCY_MHZ = 100.0
DEFAULT_CHANNEL_SPACING_MHZ = 15.0


def _rng_or_default(rng: np.random.Generator | np.random.RandomState | None) -> Any:
    """Return a supplied random generator or a fresh default generator."""

    return rng if rng is not None else np.random.default_rng()


def _validate_num_samples(num_samples: int) -> int:
    if int(num_samples) != num_samples or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    return int(num_samples)


def _validate_sample_rate(sample_rate: float) -> float:
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive finite number")
    return float(sample_rate)


def generate_awgn(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    noise_power_db: float = -20.0,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> np.ndarray:
    """Generate complex additive white Gaussian noise.

    ``noise_power_db`` is the expected complex power, i.e. ``E[|I+jQ|^2]``.
    The finite-length sample power will vary naturally around that value.
    """

    num_samples = _validate_num_samples(num_samples)
    if not np.isfinite(noise_power_db):
        raise ValueError("noise_power_db must be finite")

    random = _rng_or_default(rng)
    linear_power = 10.0 ** (float(noise_power_db) / 10.0)
    component_std = np.sqrt(linear_power / 2.0)
    i_samples = random.normal(0.0, component_std, num_samples)
    q_samples = random.normal(0.0, component_std, num_samples)
    return np.asarray(i_samples + 1j * q_samples, dtype=np.complex128)


def generate_civilian_signal(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    carrier_freq_hz: float = 50.0e3,
    snr_db: float = 15.0,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> np.ndarray:
    """Generate a continuous phase-modulated civilian transmission."""

    num_samples = _validate_num_samples(num_samples)
    sample_rate = _validate_sample_rate(sample_rate)
    if abs(carrier_freq_hz) >= sample_rate / 2.0:
        raise ValueError("carrier_freq_hz must be inside the Nyquist band")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")

    random = _rng_or_default(rng)
    t = np.arange(num_samples, dtype=float) / sample_rate
    message_freq_hz = min(1.0e3, sample_rate / 10.0)
    message = np.sin(2.0 * np.pi * message_freq_hz * t)
    carrier_amplitude = 0.8
    phase = 2.0 * np.pi * carrier_freq_hz * t + 1.5 * message
    clean_signal = carrier_amplitude * np.exp(1j * phase)
    signal_power_db = 10.0 * np.log10(carrier_amplitude**2)
    noise = generate_awgn(
        num_samples,
        noise_power_db=signal_power_db - float(snr_db),
        rng=random,
    )
    return np.asarray(clean_signal + noise, dtype=np.complex128)


def generate_hostile_radar(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    carrier_freq_hz: float = 200.0e3,
    pulse_width_sec: float = 20.0e-6,
    pri_sec: float = 100.0e-6,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> np.ndarray:
    """Generate a pulsed carrier with a 20 µs width and 100 µs PRI by default."""

    num_samples = _validate_num_samples(num_samples)
    sample_rate = _validate_sample_rate(sample_rate)
    if abs(carrier_freq_hz) >= sample_rate / 2.0:
        raise ValueError("carrier_freq_hz must be inside the Nyquist band")
    if pulse_width_sec <= 0 or pri_sec <= 0 or pulse_width_sec > pri_sec:
        raise ValueError("pulse_width_sec must be positive and no greater than pri_sec")

    random = _rng_or_default(rng)
    t = np.arange(num_samples, dtype=float) / sample_rate
    pulse_period_samples = max(1, int(round(pri_sec * sample_rate)))
    pulse_width_samples = max(1, int(round(pulse_width_sec * sample_rate)))
    pulse_mask = (np.arange(num_samples) % pulse_period_samples) < pulse_width_samples
    carrier = 3.5 * np.exp(1j * 2.0 * np.pi * carrier_freq_hz * t)
    pulse_train = carrier * pulse_mask
    baseline_noise = generate_awgn(num_samples, noise_power_db=-18.0, rng=random)
    return np.asarray(pulse_train + baseline_noise, dtype=np.complex128)


def generate_hostile_jammer(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    jammer_power_db: float = 10.0,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> np.ndarray:
    """Generate high-power broadband complex Gaussian jamming noise."""

    return generate_awgn(
        _validate_num_samples(num_samples),
        noise_power_db=jammer_power_db,
        rng=rng,
    )


def generate_spectrum_batch(
    num_channels: int = 10,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    base_freq_mhz: float = DEFAULT_BASE_FREQUENCY_MHZ,
    channel_spacing_mhz: float = DEFAULT_CHANNEL_SPACING_MHZ,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> dict[int, dict[str, Any]]:
    """Generate a multi-channel spectrum batch.

    Each channel contains ``iq``, frequency, sample-rate, and ground-truth
    metadata. For a normal ten-channel dashboard batch, all four signal
    classes are represented at least once so the UI remains useful on every
    sweep; additional channels are sampled from the configured distribution.
    """

    if int(num_channels) != num_channels or num_channels <= 0:
        raise ValueError("num_channels must be a positive integer")
    num_channels = int(num_channels)
    num_samples = _validate_num_samples(num_samples)
    sample_rate = _validate_sample_rate(sample_rate)
    if not np.isfinite(base_freq_mhz) or not np.isfinite(channel_spacing_mhz):
        raise ValueError("frequency settings must be finite")

    random = _rng_or_default(rng)
    signal_choices = np.array(["noise", "civilian", "radar", "jammer"])
    choice_probs = np.array([0.40, 0.30, 0.15, 0.15])
    if num_channels >= len(signal_choices):
        assignments = list(signal_choices)
        if num_channels > len(assignments):
            assignments.extend(
                random.choice(signal_choices, size=num_channels - len(assignments), p=choice_probs)
            )
        random.shuffle(assignments)
    else:
        assignments = list(random.choice(signal_choices, size=num_channels, p=choice_probs))

    channels: dict[int, dict[str, Any]] = {}
    for channel_id, selected_type in enumerate(assignments):
        center_freq_mhz = float(base_freq_mhz + channel_id * channel_spacing_mhz)
        if selected_type == "noise":
            iq = generate_awgn(num_samples, noise_power_db=-22.0, rng=random)
            signal_class = "Background Noise"
        elif selected_type == "civilian":
            iq = generate_civilian_signal(
                num_samples,
                sample_rate=sample_rate,
                carrier_freq_hz=float(random.uniform(20.0e3, 100.0e3)),
                snr_db=float(random.uniform(10.0, 25.0)),
                rng=random,
            )
            signal_class = "Civilian Broadcast"
        elif selected_type == "radar":
            iq = generate_hostile_radar(
                num_samples,
                sample_rate=sample_rate,
                carrier_freq_hz=float(random.uniform(150.0e3, 300.0e3)),
                pulse_width_sec=float(random.uniform(10.0e-6, 30.0e-6)),
                rng=random,
            )
            signal_class = "Hostile Radar"
        else:
            iq = generate_hostile_jammer(
                num_samples,
                jammer_power_db=float(random.uniform(5.0, 15.0)),
                rng=random,
            )
            signal_class = "Hostile Jammer"

        channels[channel_id] = {
            "iq": iq,
            "type": signal_class,
            "signal_class": signal_class,
            "ground_truth": {"signal_class": signal_class, "source": "synthetic"},
            "freq_mhz": center_freq_mhz,
            "center_freq_mhz": center_freq_mhz,
            "sample_rate": sample_rate,
        }

    return channels


if __name__ == "__main__":
    batch = generate_spectrum_batch()
    print(f"Generated {len(batch)} channels successfully.")
    for channel_id, channel in batch.items():
        power_db = 10.0 * np.log10(np.mean(np.abs(channel["iq"]) ** 2) + 1e-12)
        print(
            f"Ch {channel_id} ({channel['freq_mhz']:.1f} MHz): "
            f"{channel['type']:<20} | Power: {power_db:.2f} dB"
        )
