# Current Scope and Implementation Status

## 1. What DSAS is

DSAS is a learning and demonstration system that answers one question:

> Given a batch of simulated RF channels, which channels deserve an analyst's attention first, and why?

It does this in four stages:

```text
synthetic IQ -> power/PSD measurement -> ML classification -> ranked dashboard
```

There is no radio receiver in this version. The simulator creates complex samples mathematically, so the pipeline can be developed and demonstrated without an SDR, antenna, driver, or field data.

## 2. What is implemented now

| Component | Implemented behavior | File |
| --- | --- | --- |
| Synthetic signal generator | Creates complex AWGN, continuous phase-modulated civilian signals, pulsed radar, and broadband jammer signals. | `simulator.py` |
| Batch generator | Returns ten channels by default. Each record includes IQ, sample rate, center frequency, class metadata, and a synthetic-data marker. | `simulator.py` |
| PSD and energy scanner | Uses `scipy.signal.welch()` with a 256-sample segment. Computes mean power in dB and peak PSD. | `scanner.py` |
| Priority engine | Computes `max(0, power_db - threshold_db)` and applies CRITICAL 2.5x, HIGH 1.8x, MEDIUM 1.2x, or LOW 1.0x. | `scanner.py` |
| Classifier | Extracts nine features, scales them, and predicts one of four classes with a Random Forest. | `classifier.py` |
| Dashboard | Provides threshold control, manual sweep, auto-refresh, clear-log action, KPIs, live power chart, waterfall, threat feed, incident log, IQ waveform, and STFT spectrogram. | `app.py` |
| Dependency manifest | Lists NumPy, SciPy, scikit-learn, pandas, Plotly, and Streamlit. | `requirements.txt` |

The default batch deliberately contains all four signal classes when at least four channels are requested. That makes a demo repeatable and prevents an otherwise valid random sweep from showing no threats at all.

## 3. What happens during one scan

1. `generate_spectrum_batch()` creates ten channel records.
2. `scan_and_prioritize()` measures each IQ array with Welch PSD.
3. The classifier extracts a nine-number feature vector and returns a class, confidence, threat level, and status.
4. The scanner computes power excess and the final priority score.
5. The DataFrame is sorted from highest priority to lowest priority.
6. Streamlit renders the sorted observations and adds HIGH/CRITICAL rows to its in-memory incident log.

## 4. Backend data contracts

### 4.1 Channel record

Every batch entry is a dictionary shaped like this:

```python
{
    "iq": np.ndarray,                 # complex128, shape (1024,)
    "type": "Hostile Radar",         # ground-truth display label
    "signal_class": "Hostile Radar",
    "ground_truth": {
        "signal_class": "Hostile Radar",
        "source": "synthetic",
    },
    "freq_mhz": 130.0,
    "center_freq_mhz": 130.0,
    "sample_rate": 1_000_000.0,
}
```

Consumers must use `iq` and `sample_rate` for signal processing. Ground-truth fields are for simulation analysis and fallback behavior; a future real-input adapter must not provide them to the classifier.

### 4.2 Classifier result

`classify_channel()` returns:

```python
{
    "classification": "Hostile Jammer",
    "confidence": 98.2,
    "threat_level": "CRITICAL",
    "status": "BROADBAND JAMMING",
    "power_db": 10.1,
    "papr_db": 8.0,
    "features": np.ndarray,            # nine values
}
```

### 4.3 Scanner result

`scan_and_prioritize()` returns a pandas DataFrame with these important columns:

`Channel`, `Center Freq (MHz)`, `Power (dB)`, `Classification`, `Confidence (%)`, `Threat Level`, `Status`, `Power Excess (dB)`, `Threat Multiplier`, `Priority Score`, and `Timestamp`.

The score is intentionally explainable:

```text
priority_score = max(0, power_db - threshold_db) * threat_multiplier
```

## 5. The nine classifier features

The current feature order is fixed and must not be changed without updating both training and prediction code:

1. Mean power in dB.
2. Peak-to-average power ratio in dB.
3. Amplitude standard deviation.
4. Amplitude kurtosis.
5. Phase standard deviation.
6. Instantaneous-frequency standard deviation.
7. Spectral flatness.
8. Spectral centroid.
9. Spectral spread.

The public `train(samples_per_class=300)` method generates all training features in memory. The shared dashboard classifier uses a smaller first-use training set to reduce initial UI latency. No model file is required for normal operation; explicit save/load uses a local pickle file.

## 6. Current user-visible behavior

### Sidebar

- Noise-floor threshold slider from -25 dB to +10 dB.
- Manual sweep button.
- Auto-refresh toggle and interval.
- Clear threat-log button.
- Classifier/fallback status.

### Live Spectrum Monitor

- Bar chart of aggregate power per center-frequency channel.
- Threshold line.
- Short power-history waterfall.

The bar chart is channel energy, not a continuous wideband spectrum. A true frequency-bin spectrum is a recommended next feature.

### Prioritized Threat Feed

- Sorted observations.
- Row coloring for threat severity.
- Persistent HIGH/CRITICAL incident list for the current Streamlit session.

### Signal Inspector

- Channel selector.
- I/Q time-domain traces.
- Two-sided STFT spectrogram.

## 7. Explicit non-goals

The current scope excludes:

- RTL-SDR, HackRF, USRP, or any other hardware integration.
- C bindings, SDR drivers, network capture, or live RF collection.
- Deep neural networks, GPU training, or long-running optimization.
- Real-world threat attribution or operational decisions.
- Distributed sensors, authentication, multi-user collaboration, or a production database.

## 8. Known gaps to work on next

- There is no committed `tests/` suite yet; current validation is smoke and Streamlit harness testing.
- There are no measured validation metrics such as a held-out confusion matrix or per-class recall.
- The model is trained on synthetic distributions and is not calibrated for field data.
- The incident log is not durable and has no CSV/JSON export.
- The waterfall keeps aggregate channel power, not raw PSD history.
- Threshold changes re-score the current batch but do not maintain a calibrated noise-floor estimator.
- The application emits a current Streamlit deprecation warning for `use_container_width`.
- There is no CI pipeline, type checking, linting, or reproducible release process.

These gaps define the next work rather than indicating that SDR hardware should be added immediately.
