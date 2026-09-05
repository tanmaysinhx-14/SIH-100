"""
classifier.py - Signal Feature Extraction and Machine Learning Threat Classifier
Extracts statistical and spectral metrics from IQ signals and classifies them
using a trained Random Forest model.
"""

import os
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Fallback simulator import if available
try:
    from simulator import (
        generate_awgn,
        generate_civilian_signal,
        generate_hostile_radar,
        generate_hostile_jammer
    )
    HAS_SIMULATOR = True
except ImportError:
    HAS_SIMULATOR = False

MODEL_FILE = "rf_signal_model.joblib"


# ==========================================
# 1. FEATURE EXTRACTION ENGINE
# ==========================================
def extract_signal_features(iq_samples: np.ndarray, sample_rate: float = 1e6) -> np.ndarray:
    """
    Extracts a 9-dimensional statistical and spectral feature vector from complex IQ samples.
    
    Features:
    1. Mean Power (dB)
    2. Peak-to-Average Power Ratio (PAPR in dB)
    3. Amplitude Standard Deviation
    4. Amplitude Kurtosis (peakedness / pulse indicator)
    5. Phase Standard Deviation
    6. Instantaneous Frequency Standard Deviation
    7. Spectral Flatness (distinguishes broadband noise/jamming from narrow carriers)
    8. Spectral Centroid
    9. Spectral Spread (Bandwidth indicator)
    """
    amp = np.abs(iq_samples)
    power = amp ** 2
    mean_power = np.mean(power) + 1e-12
    peak_power = np.max(power) + 1e-12
    
    # 1. Power & PAPR
    mean_power_db = 10 * np.log10(mean_power)
    papr_db = 10 * np.log10(peak_power / mean_power)
    
    # 2. Amplitude statistics
    amp_std = np.std(amp)
    amp_kurtosis = stats.kurtosis(amp)
    
    # 3. Phase & Frequency statistics
    phase = np.angle(iq_samples)
    phase_std = np.std(phase)
    
    # Instantaneous frequency (derivative of unwrapped phase)
    unwrapped_phase = np.unwrap(phase)
    inst_freq = np.diff(unwrapped_phase) * sample_rate / (2 * np.pi)
    freq_std = np.std(inst_freq)
    
    # 4. Spectral statistics (via Periodogram)
    freqs, psd = signal.periodogram(iq_samples, fs=sample_rate, return_onesided=False)
    psd = psd + 1e-12  # Avoid division by zero
    
    # Spectral Flatness = Geometric Mean / Arithmetic Mean
    geometric_mean = np.exp(np.mean(np.log(psd)))
    arithmetic_mean = np.mean(psd)
    spectral_flatness = geometric_mean / arithmetic_mean
    
    # Spectral Centroid & Spread
    psd_norm = psd / np.sum(psd)
    spectral_centroid = np.sum(freqs * psd_norm)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))
    
    return np.array([
        mean_power_db,
        papr_db,
        amp_std,
        amp_kurtosis,
        phase_std,
        freq_std,
        spectral_flatness,
        spectral_centroid,
        spectral_spread
    ])


# ==========================================
# 2. SYNTHETIC TRAINING DATA GENERATOR
# ==========================================
def _generate_synthetic_training_data(samples_per_class: int = 300, num_iq_samples: int = 1024):
    """Generates labeled feature vectors for model training."""
    X = []
    y = []
    
    classes = ["Background Noise", "Civilian Broadcast", "Hostile Radar", "Hostile Jammer"]
    
    for _ in range(samples_per_class):
        # 1. Background Noise
        if HAS_SIMULATOR:
            iq_noise = generate_awgn(num_iq_samples, noise_power_db=np.random.uniform(-25, -15))
        else:
            p = np.random.uniform(-25, -15)
            iq_noise = (np.random.randn(num_iq_samples) + 1j * np.random.randn(num_iq_samples)) * np.sqrt(10**(p/10)/2)
        X.append(extract_signal_features(iq_noise))
        y.append("Background Noise")
        
        # 2. Civilian Broadcast
        if HAS_SIMULATOR:
            iq_civ, _ = generate_civilian_signal(
                num_iq_samples, 
                carrier_freq_hz=np.random.uniform(20e3, 100e3),
                snr_db=np.random.uniform(10, 25)
            )
        else:
            t = np.arange(num_iq_samples) / 1e6
            iq_civ = np.exp(1j * 2 * np.pi * 50e3 * t) * 0.8 + generate_awgn(num_iq_samples, -20) if HAS_SIMULATOR else np.exp(1j * 2 * np.pi * 50e3 * t)
        X.append(extract_signal_features(iq_civ))
        y.append("Civilian Broadcast")
        
        # 3. Hostile Radar
        if HAS_SIMULATOR:
            iq_radar, _ = generate_hostile_radar(
                num_iq_samples,
                carrier_freq_hz=np.random.uniform(150e3, 300e3),
                pulse_width_sec=np.random.uniform(10e-6, 30e-6)
            )
        else:
            t = np.arange(num_iq_samples) / 1e6
            mask = (t * 1e4).astype(int) % 10 < 3
            iq_radar = np.exp(1j * 2 * np.pi * 200e3 * t) * mask * 3.5
        X.append(extract_signal_features(iq_radar))
        y.append("Hostile Radar")
        
        # 4. Hostile Jammer
        if HAS_SIMULATOR:
            iq_jammer, _ = generate_hostile_jammer(num_iq_samples, jammer_power_db=np.random.uniform(5, 15))
        else:
            iq_jammer = (np.random.randn(num_iq_samples) + 1j * np.random.randn(num_iq_samples)) * 2.5
        X.append(extract_signal_features(iq_jammer))
        y.append("Hostile Jammer")
        
    return np.array(X), np.array(y)


# ==========================================
# 3. CLASSIFIER MODEL PIPELINE
# ==========================================
class RFSignalClassifier:
    """Random Forest Classifier wrapper with auto-training capabilities."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.is_trained = False
        
    def train(self, samples_per_class: int = 300):
        """Trains the Random Forest model on generated synthetic dataset."""
        X, y = _generate_synthetic_training_data(samples_per_class=samples_per_class)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
    def save(self, filepath: str = MODEL_FILE):
        """Saves trained model and scaler to disk."""
        joblib.dump({"model": self.model, "scaler": self.scaler}, filepath)
        
    def load(self, filepath: str = MODEL_FILE) -> bool:
        """Loads trained model and scaler from disk if exists."""
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_trained = True
            return True
        return False

    def predict(self, iq_samples: np.ndarray, sample_rate: float = 1e6) -> dict:
        """
        Classifies an IQ signal array.
        Returns prediction, probability/confidence score, threat level, and operational status.
        """
        if not self.is_trained:
            if not self.load():
                self.train()
                self.save()
                
        features = extract_signal_features(iq_samples, sample_rate).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        pred_class = self.model.predict(features_scaled)[0]
        probs = self.model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probs))
        
        # Map threat severity & operational status
        if pred_class == "Hostile Jammer":
            threat_level = "CRITICAL"
            status = "BROADBAND JAMMING"
        elif pred_class == "Hostile Radar":
            threat_level = "HIGH"
            status = "PULSED RADAR LOCK"
        elif pred_class == "Civilian Broadcast":
            threat_level = "LOW"
            status = "STANDARD COMM"
        else:
            threat_level = "LOW"
            status = "CLEAR"
            
        return {
            "classification": pred_class,
            "confidence": round(confidence * 100, 2),
            "threat_level": threat_level,
            "status": status,
            "power_db": round(features[0][0], 2),
            "papr_db": round(features[0][1], 2)
        }


# Singleton model instance
_CLASSIFIER_INSTANCE = RFSignalClassifier()


def classify_channel(iq_samples: np.ndarray, sample_rate: float = 1e6) -> dict:
    """Convenience function for app.py / scanner.py integration."""
    return _CLASSIFIER_INSTANCE.predict(iq_samples, sample_rate)


if __name__ == "__main__":
    print("Initializing and training Random Forest Signal Classifier...")
    clf = RFSignalClassifier()
    clf.train(samples_per_class=400)
    clf.save()
    print("Model trained and saved successfully.")
    
    # Test on a dummy pulse wave
    t = np.arange(1024) / 1e6
    mask = (t * 1e4).astype(int) % 10 < 3
    test_iq = np.exp(1j * 2 * np.pi * 200e3 * t) * mask * 3.5 + 0.1 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    result = classify_channel(test_iq)
    print("\nTest Prediction Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")