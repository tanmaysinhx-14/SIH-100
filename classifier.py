"""Fast feature extraction and Random Forest classification for DSAS signals."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from simulator import (
    generate_awgn,
    generate_civilian_signal,
    generate_hostile_jammer,
    generate_hostile_radar,
)


MODEL_FILE = "rf_signal_model.pkl"
FEATURE_NAMES = (
    "Mean Power (dB)",
    "PAPR (dB)",
    "Amplitude Std",
    "Amplitude Kurtosis",
    "Phase Std",
    "Instantaneous Frequency Std",
    "Spectral Flatness",
    "Spectral Centroid",
    "Spectral Spread",
)
CLASS_LABELS = (
    "Background Noise",
    "Civilian Broadcast",
    "Hostile Radar",
    "Hostile Jammer",
)


def _validated_iq(iq_samples: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq_samples, dtype=np.complex128).reshape(-1)
    if iq.size < 2:
        raise ValueError("iq_samples must contain at least two samples")
    if not np.all(np.isfinite(iq)):
        raise ValueError("iq_samples must contain only finite values")
    return iq


def _finite(value: float, default: float = 0.0) -> float:
    value = float(value)
    return value if np.isfinite(value) else default


def extract_signal_features(
    iq_samples: np.ndarray,
    sample_rate: float = 1.0e6,
) -> np.ndarray:
    """Return the required nine-dimensional feature vector for one IQ signal."""

    iq = _validated_iq(iq_samples)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive finite number")

    amplitude = np.abs(iq)
    power = amplitude**2
    mean_power = max(float(np.mean(power)), np.finfo(float).tiny)
    peak_power = max(float(np.max(power)), np.finfo(float).tiny)

    mean_power_db = 10.0 * np.log10(mean_power)
    papr_db = 10.0 * np.log10(peak_power / mean_power)
    amplitude_std = float(np.std(amplitude))
    amplitude_kurtosis = _finite(stats.kurtosis(amplitude, fisher=False, bias=False))

    phase = np.angle(iq)
    phase_std = float(np.std(phase))
    unwrapped_phase = np.unwrap(phase)
    instantaneous_frequency = np.diff(unwrapped_phase) * float(sample_rate) / (2.0 * np.pi)
    instantaneous_frequency_std = float(np.std(instantaneous_frequency))

    # A two-sided periodogram is appropriate for complex baseband IQ data.
    frequencies, psd = signal.periodogram(
        iq,
        fs=float(sample_rate),
        window="hann",
        detrend=False,
        return_onesided=False,
        scaling="density",
    )
    frequencies = np.fft.fftshift(frequencies)
    psd = np.fft.fftshift(np.maximum(np.real(psd), 0.0))
    psd_floor = np.finfo(float).tiny
    psd_safe = np.maximum(psd, psd_floor)
    geometric_mean = np.exp(np.mean(np.log(psd_safe)))
    arithmetic_mean = max(float(np.mean(psd_safe)), psd_floor)
    spectral_flatness = _finite(geometric_mean / arithmetic_mean)
    psd_sum = float(np.sum(psd_safe))
    psd_weights = psd_safe / psd_sum
    spectral_centroid = _finite(np.sum(frequencies * psd_weights))
    spectral_spread = _finite(
        np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * psd_weights))
    )

    return np.asarray(
        [
            mean_power_db,
            papr_db,
            amplitude_std,
            amplitude_kurtosis,
            phase_std,
            instantaneous_frequency_std,
            spectral_flatness,
            spectral_centroid,
            spectral_spread,
        ],
        dtype=float,
    )


def _generate_synthetic_training_data(
    samples_per_class: int = 300,
    num_iq_samples: int = 1024,
    random_state: int = 2024,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate labeled features entirely in memory for model training."""

    if int(samples_per_class) != samples_per_class or samples_per_class < 1:
        raise ValueError("samples_per_class must be a positive integer")
    if int(num_iq_samples) != num_iq_samples or num_iq_samples < 2:
        raise ValueError("num_iq_samples must be at least two")

    random = np.random.default_rng(random_state)
    features: list[np.ndarray] = []
    labels: list[str] = []
    for _ in range(int(samples_per_class)):
        noise = generate_awgn(
            num_iq_samples,
            noise_power_db=float(random.uniform(-25.0, -15.0)),
            rng=random,
        )
        features.append(extract_signal_features(noise))
        labels.append("Background Noise")

        civilian = generate_civilian_signal(
            num_iq_samples,
            carrier_freq_hz=float(random.uniform(20.0e3, 100.0e3)),
            snr_db=float(random.uniform(10.0, 25.0)),
            rng=random,
        )
        features.append(extract_signal_features(civilian))
        labels.append("Civilian Broadcast")

        radar = generate_hostile_radar(
            num_iq_samples,
            carrier_freq_hz=float(random.uniform(150.0e3, 300.0e3)),
            pulse_width_sec=float(random.uniform(10.0e-6, 30.0e-6)),
            rng=random,
        )
        features.append(extract_signal_features(radar))
        labels.append("Hostile Radar")

        jammer = generate_hostile_jammer(
            num_iq_samples,
            jammer_power_db=float(random.uniform(5.0, 15.0)),
            rng=random,
        )
        features.append(extract_signal_features(jammer))
        labels.append("Hostile Jammer")

    return np.vstack(features), np.asarray(labels, dtype=object)


class RFSignalClassifier:
    """Standard-scaled, shallow Random Forest classifier for the four classes."""

    def __init__(self, random_state: int = 42) -> None:
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_trained = False

    def train(self, samples_per_class: int = 300) -> "RFSignalClassifier":
        """Generate synthetic features and train the model, returning ``self``."""

        x_train, labels = _generate_synthetic_training_data(samples_per_class)
        self.scaler.fit(x_train)
        self.model.fit(self.scaler.transform(x_train), labels)
        self.is_trained = True
        return self

    def save(self, filepath: str | Path = MODEL_FILE) -> Path:
        """Persist the trained model and scaler using Python's standard library."""

        if not self.is_trained:
            raise RuntimeError("train the classifier before saving it")
        destination = Path(filepath)
        with destination.open("wb") as handle:
            pickle.dump({"model": self.model, "scaler": self.scaler}, handle)
        return destination

    def load(self, filepath: str | Path = MODEL_FILE) -> bool:
        """Load a previously saved model; return ``False`` when it is absent."""

        source = Path(filepath)
        if not source.exists():
            return False
        with source.open("rb") as handle:
            payload: dict[str, Any] = pickle.load(handle)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.is_trained = True
        return True

    def predict(
        self,
        iq_samples: np.ndarray,
        sample_rate: float = 1.0e6,
    ) -> dict[str, Any]:
        """Classify one IQ signal and return label, confidence, and severity."""

        if not self.is_trained:
            # Keep first-use dashboard latency low while the public train()
            # method still follows the requested 300-samples-per-class default.
            self.train(samples_per_class=100)

        feature_vector = extract_signal_features(iq_samples, sample_rate)
        scaled_features = self.scaler.transform(feature_vector.reshape(1, -1))
        prediction = str(self.model.predict(scaled_features)[0])
        probabilities = self.model.predict_proba(scaled_features)[0]
        confidence = float(np.max(probabilities))

        severity = {
            "Hostile Jammer": ("CRITICAL", "BROADBAND JAMMING"),
            "Hostile Radar": ("HIGH", "PULSED RADAR LOCK"),
            "Civilian Broadcast": ("LOW", "STANDARD COMM"),
            "Background Noise": ("LOW", "CLEAR"),
        }
        threat_level, status = severity.get(prediction, ("MEDIUM", "ANOMALOUS ACTIVITY"))
        return {
            "classification": prediction,
            "confidence": round(confidence * 100.0, 2),
            "threat_level": threat_level,
            "status": status,
            "power_db": round(float(feature_vector[0]), 2),
            "papr_db": round(float(feature_vector[1]), 2),
            "features": feature_vector,
        }


_CLASSIFIER_INSTANCE = RFSignalClassifier()


def classify_channel(iq_samples: np.ndarray, sample_rate: float = 1.0e6) -> dict[str, Any]:
    """Classify a channel with the process-local shared model instance."""

    return _CLASSIFIER_INSTANCE.predict(iq_samples, sample_rate)


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    classifier = RFSignalClassifier().train(samples_per_class=300)
    elapsed = time.perf_counter() - start
    print(f"Trained Random Forest in {elapsed:.2f}s")
    test_signal = generate_hostile_radar()
    print(classifier.predict(test_signal))
