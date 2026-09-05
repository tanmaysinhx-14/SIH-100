"""
simulator.py - Synthetic RF Signal Generation Engine
Generates complex IQ time-series data for simulated electromagnetic spectrum channels.
"""

import numpy as np


def generate_awgn(num_samples: int, noise_power_db: float = -20.0) -> np.ndarray:
    """Generates Additive White Gaussian Noise (AWGN) as complex IQ samples."""
    linear_power = 10.0 ** (noise_power_db / 10.0)
    std_dev = np.sqrt(linear_power / 2.0)
    
    i_samples = np.random.normal(0, std_dev, num_samples)
    q_samples = np.random.normal(0, std_dev, num_samples)
    return i_samples + 1j * q_samples


def generate_civilian_signal(
    num_samples: int, 
    sample_rate: float = 1e6, 
    carrier_freq_hz: float = 50e3,
    snr_db: float = 15.0
) -> tuple[np.ndarray, str]:
    """
    Simulates a continuous civilian broadcast (e.g., FM audio or continuous tone)
    with phase modulation over an AWGN background.
    """
    t = np.arange(num_samples) / sample_rate
    message = np.sin(2 * np.pi * 1e3 * t)  # 1 kHz modulating tone
    
    # Phase modulation
    mod_index = 1.5
    iq_signal = 0.8 * np.exp(1j * (2 * np.pi * carrier_freq_hz * t + mod_index * message))
    
    # Add noise
    noise_power_db = 10 * np.log10(0.8**2) - snr_db
    noise = generate_awgn(num_samples, noise_power_db=noise_power_db)
    
    return iq_signal + noise, "Civilian Broadcast"


def generate_hostile_radar(
    num_samples: int, 
    sample_rate: float = 1e6, 
    carrier_freq_hz: float = 200e3,
    pulse_width_sec: float = 20e-6,
    pri_sec: float = 100e-6
) -> tuple[np.ndarray, str]:
    """
    Simulates a high-power hostile targeting or tracking radar pulse train.
    High peak-to-average power ratio with distinct pulse repetition intervals.
    """
    t = np.arange(num_samples) / sample_rate
    
    # Create periodic pulse train mask
    pulse_period_samples = int(pri_sec * sample_rate)
    pulse_width_samples = int(pulse_width_sec * sample_rate)
    
    mask = (np.arange(num_samples) % pulse_period_samples) < pulse_width_samples
    
    # Pulsed complex carrier
    radar_pulse = 3.5 * np.exp(1j * 2 * np.pi * carrier_freq_hz * t) * mask
    
    # Add baseline noise
    noise = generate_awgn(num_samples, noise_power_db=-18.0)
    
    return radar_pulse + noise, "Hostile Radar"


def generate_hostile_jammer(
    num_samples: int, 
    jammer_power_db: float = 10.0
) -> tuple[np.ndarray, str]:
    """
    Simulates high-power broadband electronic jamming designed to deny spectrum access.
    Massive broadband noise floor elevation across the channel.
    """
    # High amplitude broadband Gaussian noise across the entire channel bandwidth
    jammer_iq = generate_awgn(num_samples, noise_power_db=jammer_power_db)
    return jammer_iq, "Hostile Jammer"


def generate_spectrum_batch(
    num_channels: int = 10,
    num_samples: int = 1024,
    sample_rate: float = 1e6,
    base_freq_mhz: float = 100.0,
    channel_spacing_mhz: float = 15.0
) -> dict:
    """
    Generates a full batch of simulated RF spectrum channels with randomized signal conditions.
    
    Returns a dictionary structured for scanner.py and app.py consumption:
    {
        channel_index: {
            "iq": np.ndarray (complex128),
            "type": str,
            "freq_mhz": float,
            "sample_rate": float
        }
    }
    """
    channels = {}
    
    # Probability weights for signal assignment across channels
    # 40% Noise, 30% Civilian, 15% Radar, 15% Jammer
    signal_choices = ["noise", "civilian", "radar", "jammer"]
    choice_probs = [0.40, 0.30, 0.15, 0.15]
    
    for ch in range(num_channels):
        center_freq = base_freq_mhz + (ch * channel_spacing_mhz)
        selected_type = np.random.choice(signal_choices, p=choice_probs)
        
        if selected_type == "noise":
            iq = generate_awgn(num_samples, noise_power_db=-22.0)
            sig_type = "Background Noise"
        elif selected_type == "civilian":
            iq, sig_type = generate_civilian_signal(num_samples, sample_rate)
        elif selected_type == "radar":
            iq, sig_type = generate_hostile_radar(num_samples, sample_rate)
        else:
            iq, sig_type = generate_hostile_jammer(num_samples, jammer_power_db=8.0)
            
        channels[ch] = {
            "iq": iq,
            "type": sig_type,
            "freq_mhz": center_freq,
            "sample_rate": sample_rate
        }
        
    return channels


if __name__ == "__main__":
    # Test execution
    batch = generate_spectrum_batch()
    print(f"Generated {len(batch)} channels successfully.")
    for ch_idx, ch_info in batch.items():
        power = np.mean(np.abs(ch_info['iq'])**2)
        power_db = 10 * np.log10(power)
        print(f"Ch {ch_idx} ({ch_info['freq_mhz']} MHz): {ch_info['type']:<20} | Power: {power_db:.2f} dB")