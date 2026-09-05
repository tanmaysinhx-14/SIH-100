# Problem Statement: Defensive Spectrum-Awareness System (4.5-Hour MVP)

## Overview
Modern electronic warfare and defensive operations require real-time monitoring of the electromagnetic (EM) spectrum to detect hostile threats such as radar locks and electronic jamming. Due to the vastness of the EM spectrum, manual continuous scanning is impossible. 

## Core Objective
Develop an automated, software-defined defensive spectrum-awareness prototype in Python that:
1. Simulates multi-channel radio frequency (RF) spectrum data with mixed noise, civilian signals, and hostile signatures.
2. Intelligently prioritizes scanning toward frequency bands displaying high energy or anomalous activity.
3. Classifies detected signals into threat categories using signal processing and machine learning.
4. Visualizes live spectrum sweeps and prioritized threat alerts on an interactive dashboard.

## 4.5-Hour MVP Scope & Boundaries
To deliver within the 4.5-hour timeframe, the project is strictly scoped as follows:

### In-Scope (Must Build)
- **Synthetic RF Generator:** Programmatic generation of 10 distinct frequency channels containing Additive White Gaussian Noise (AWGN), continuous civilian communication (Wi-Fi/FM), hostile radar pulses, and broadband jamming.
- **Smart Priority Scanner:** Energy Detection (FFT Power Spectral Density) algorithm that ranks channels by activity level to scan high-probability threat bands first.
- **Signal Classifier:** Feature extraction pipeline (Peak Power, Bandwidth, Spectral Flatness, Pulse Repetition) paired with a fast Random Forest / Lightweight CNN model trained instantly on synthetic data.
- **Interactive GUI:** A Streamlit dashboard displaying:
  - Real-time Spectrum Visualizer (FFT Plot / Heatmap using Plotly).
  - Prioritized Threat Feed (High / Medium / Low urgency table).
  - Signal Inspector (Click to view IQ waveform and spectrogram).

### Out-of-Scope (Deferred)
- Live Software-Defined Radio (SDR) hardware integration (HackRF/RTL-SDR).
- Heavy deep learning training loops (over 2 minutes train time).
- Multi-node distributed sensor networks.