"""Centralized configuration parameters for the DSAS spectrum awareness system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class SystemConfig:
    """System configuration parameters."""

    # Signal Defaults
    default_num_samples: int = 1024
    default_sample_rate: float = 1.0e6
    default_base_frequency_mhz: float = 100.0
    default_channel_spacing_mhz: float = 15.0

    # DSP Defaults
    welch_nperseg: int = 256
    default_noise_threshold_db: float = -10.0

    # Prioritization Multipliers
    threat_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "CRITICAL": 2.5,
            "HIGH": 1.8,
            "MEDIUM": 1.2,
            "LOW": 1.0,
        }
    )

    # Feature Metadata
    feature_names: Tuple[str, ...] = (
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

    class_labels: Tuple[str, ...] = (
        "Background Noise",
        "Civilian Broadcast",
        "Hostile Radar",
        "Hostile Jammer",
    )


DEFAULT_CONFIG = SystemConfig()
