# System Architecture

## Core Pipeline Architecture

[ Synthetic RF Generator ] 
          │ (Raw IQ Signals / Time Series)
          ▼
[ Priority Scanning Engine ] ──► (FFT Energy Thresholding & Priority Queue)
          │
          ▼
[ Feature Extractor & STFT ] ──► (Spectrograms & Spectral Metrics)
          │
          ▼
[ ML Classifier Engine ]     ──► (Class Prediction & Confidence Score)
          │
          ▼
[ Interactive Streamlit GUI ]──► (Plotly Charts, Threat Table, Signal Inspector)

## Module Responsibilities

1. `simulator.py`
   - Generates complex IQ data ($I + jQ$) across $N$ simulated channels.
   - Injects 3 signal classes: Civilian (OFDM/AM), Hostile Radar (Pulsed), Hostile Jammer (Broadband Noise).

2. `scanner.py`
   - Performs fast Fast Fourier Transform (FFT) power calculation across all channels.
   - Implements a priority queue prioritizing channels exceeding baseline noise energy thresholds.

3. `classifier.py`
   - Computes Short-Time Fourier Transform (STFT) spectrogram features and spectral statistics.
   - Runs a lightweight pretrained or fast-trained classifier returning label (`Hostile Radar`, `Hostile Jammer`, `Civilian`, `Background Noise`) and threat severity score.

4. `app.py` (Dashboard)
   - Streamlit application serving as the UI.
   - Automatically polls the scanner/classifier and updates Plotly visual elements.