# Defensive Spectrum-Awareness System (DSAS)

## Comprehensive Project Abstract and Beginner's Guide

**Project type:** Software-only defensive RF spectrum-awareness prototype  
**Primary language:** Python 3.10+  
**Current input:** Synthetic complex IQ data only  
**Default scan:** 10 channels, 1,024 samples per channel, 1 MHz sample rate  
**Dashboard:** Streamlit and Plotly

> This file is the single, self-contained explanation of the project. It explains the motivation, the scientific ideas, the code, the data structures, the machine-learning process, the dashboard, the complete scope, and the next work in beginner-friendly language.

---

## 1. Abstract

Monitoring many radio-frequency (RF) channels manually is difficult because an analyst must inspect a large amount of signal data and decide which observations deserve attention first. The Defensive Spectrum-Awareness System (DSAS) is a hardware-free prototype that demonstrates how software can help with that first decision.

DSAS creates simulated complex in-phase and quadrature (IQ) samples for several channels. It measures the energy in each channel using a power spectral density (PSD) calculation, compares the measured power with a configurable noise-floor threshold, classifies the signal with a lightweight Random Forest model, and combines the energy and threat category into an explainable priority score. A Streamlit dashboard then shows the result as a live spectrum view, a prioritized threat feed, an incident log, an IQ waveform, and a short-time Fourier transform (STFT) spectrogram.

The system is a learning, testing, and demonstration tool. It is not a validated detector for real RF environments. It does not connect to a radio, collect live transmissions, identify real emitters, or prove that a signal is hostile.

### One-sentence description

> DSAS turns simulated RF samples into a ranked list of channels that an analyst should inspect first, while showing the measurements and model output behind each rank.

---

## 2. The project in plain language

Imagine ten radio channels as ten rooms. Each room contains a sound recording, but the recording is stored as numerical samples rather than as audio. Some rooms contain only quiet background noise, some contain an ordinary continuous transmission, one may contain short repeating bursts, and one may contain loud broadband noise.

DSAS performs the following actions:

1. It creates a recording for every room.
2. It measures how much energy is in each recording.
3. It examines the shape of each recording and predicts its class.
4. It gives more urgency to signals that are both energetic and threatening.
5. It puts the highest-priority rooms at the top of an analyst's list.
6. It lets the analyst inspect the raw numerical waveform and a picture of how frequencies change over time.

The application is therefore a complete miniature pipeline:

```text
generate -> measure -> extract features -> classify -> prioritize -> visualize
```

---

## 3. What problem the project addresses

### 3.1 The practical problem

The electromagnetic spectrum contains many signals at the same time. A monitoring system may see quiet channels, normal communications, short pulses, interference, and unknown activity. Looking at every channel with equal attention is inefficient.

The first useful automation step is not to make a final operational decision. It is to answer:

> Which channels appear most important, and what measurable evidence supports that ranking?

### 3.2 The software solution

DSAS separates the problem into small stages:

- **Simulation:** create known signal patterns without hardware.
- **Measurement:** calculate power and spectral information.
- **Classification:** use signal features to predict a class.
- **Prioritization:** use a transparent formula to rank channels.
- **Presentation:** show the evidence in a dashboard.

Separating the stages is important. It means that a future input source, such as a recorded file, can replace the simulator without requiring a complete rewrite of the scanner or dashboard.

---

## 4. Full project scope

### 4.1 Included in the current MVP

- Synthetic generation of four signal classes:
  - Background noise.
  - Civilian broadcast.
  - Hostile radar pulse train.
  - Hostile broadband jammer.
- Ten simulated channels by default.
- Complex IQ arrays with 1,024 samples at a 1 MHz sample rate.
- Welch PSD calculation with a 256-sample segment.
- Mean channel power in decibels.
- Configurable noise-floor threshold, default `-10 dB`.
- Nine statistical and spectral classifier features.
- `StandardScaler` plus a 100-estimator Random Forest with maximum depth 10.
- In-memory synthetic training data.
- Threat severity mapping and an explainable priority score.
- Streamlit dashboard with:
  - Noise-floor slider.
  - Manual sweep.
  - Auto-refresh option.
  - Threat-log clearing.
  - Four KPI cards.
  - Channel-power bar chart.
  - Bounded power waterfall.
  - Styled prioritized feed.
  - Session incident log.
  - I/Q waveform.
  - STFT spectrogram.
- A heuristic fallback path when the classifier dependency is unavailable.

### 4.2 Explicitly excluded

- RTL-SDR, HackRF, USRP, or any other radio hardware.
- Antennas, live RF collection, network packet capture, or SDR drivers.
- C bindings or hardware-specific libraries.
- Deep neural networks, GPU training, or long training jobs.
- Real-world emitter attribution or geolocation.
- Automatic electronic-warfare action or transmission.
- Distributed sensor networks.
- Authentication, multi-user permissions, or a production database.
- Claims that model confidence is a calibrated probability.

### 4.3 What success means

The MVP succeeds when a beginner can install it, start the dashboard, observe the four simulated signal families, change the threshold, execute a sweep, understand why a channel is highly ranked, inspect its waveform and spectrogram, clear the incident log, and reproduce the verification checks described in this file.

---

## 5. Important RF and signal-processing ideas

This section explains the terms that appear in the code and dashboard.

### 5.1 Electromagnetic spectrum and RF

The electromagnetic spectrum is the complete range of electromagnetic waves. Radio frequency, or RF, is the part used for radio communication, radar, wireless networks, navigation, and many other systems.

An RF signal can be described by its frequency, its strength, and the way it changes over time. DSAS does not model the entire real-world spectrum. It models a small set of independent channels so the analysis process is easy to see.

### 5.2 A channel

A **channel** is one frequency region being monitored. In this prototype, every channel has:

- A numeric channel ID such as `0` or `5`.
- A center frequency in MHz, such as `100.0 MHz`.
- A sample rate, normally `1,000,000 samples/second`.
- A short sequence of complex IQ samples.
- Synthetic ground-truth metadata.

The channels are separated in the metadata by 15 MHz by default. The IQ arrays themselves are baseband-style samples used to represent the content inside each channel.

### 5.3 Analog signals and digital samples

An actual radio signal varies continuously. A computer cannot store an infinitely detailed continuous curve, so it takes measurements at regular time intervals. Each measurement is called a **sample**.

With a sample rate of 1 MHz, the simulator takes 1,000,000 measurements per second. With 1,024 samples, one simulated record lasts:

```text
1,024 / 1,000,000 seconds = 1.024 milliseconds
```

### 5.4 IQ data

IQ means **in-phase and quadrature**. Instead of storing only one number for each sample, a radio-style baseband representation stores two related numbers:

- `I`: the in-phase component.
- `Q`: the quadrature component, shifted by 90 degrees.

The code combines them into a complex number:

```text
IQ = I + jQ
```

Here, `j` is the imaginary unit. A complex IQ sample is not an error or an unusual data type. It is a compact way to retain both signal magnitude and signal phase.

For a complex sample `z = I + jQ`:

```text
amplitude = abs(z) = sqrt(I^2 + Q^2)
phase     = angle(z) = atan2(Q, I)
```

Amplitude tells us how strong the sample is. Phase tells us where the sample lies in its rotation around the complex plane. Together, they preserve more information than a single real-valued sample.

### 5.5 Baseband and carrier frequency

A **carrier** is the repeating oscillation around which information is transmitted. A radio receiver often shifts a signal down to a simpler reference range called **baseband**. The simulator creates baseband-like IQ records while allowing a carrier frequency parameter inside the sampled bandwidth.

This is a mathematical demonstration, not a complete model of an RF front end, mixer, antenna, or receiver calibration chain.

### 5.6 Nyquist limit

The sample rate limits which frequencies can be represented correctly. The highest unambiguous frequency is approximately half the sample rate:

```text
Nyquist frequency = sample_rate / 2
```

At a 1 MHz sample rate, the usable baseband range is approximately `-500 kHz` to `+500 kHz`. The simulator validates that carrier frequencies stay inside this range.

### 5.7 Power

For an IQ sample, instantaneous power is proportional to the squared magnitude:

```text
instantaneous_power = abs(IQ)^2
```

The channel's average power is the mean of that value over all samples:

```text
mean_power = mean(abs(IQ)^2)
```

### 5.8 Decibels

Decibels, written as dB, are a logarithmic way to express ratios. They are useful because signal powers can vary across a very large range.

For power, DSAS uses:

```text
power_db = 10 * log10(mean_power)
```

The number is relative to the simulator's numerical power reference. It should not be interpreted as a calibrated physical dBm measurement.

### 5.9 Noise floor

The **noise floor** is the typical background level below which activity may be insignificant. DSAS starts with a configurable threshold of `-10 dB`.

If a channel has `-13 dB` power, it is below the default threshold. If it has `-3 dB`, it is 7 dB above the threshold.

The threshold is a decision setting, not a magical property of the environment. Changing it changes the ranking calculation for the current batch.

### 5.10 Additive white Gaussian noise (AWGN)

AWGN is a common mathematical noise model:

- **Additive:** noise is added to another signal.
- **White:** its energy is spread broadly across frequency in the idealized model.
- **Gaussian:** the random values follow a normal distribution.

The simulator creates independent random I and Q values with equal variance. A requested power in dB is converted back to a linear power before the random samples are generated.

### 5.11 Signal-to-noise ratio (SNR)

SNR compares signal power with noise power. A larger SNR means the signal is easier to distinguish from the noise.

The civilian generator uses a default 15 dB SNR. The training and batch generators vary the SNR to make the classifier see slightly different examples instead of learning one exact waveform.

### 5.12 FFT

The fast Fourier transform, or FFT, converts a signal from the time domain into a frequency-domain representation.

- **Time domain:** how the sample value changes over time.
- **Frequency domain:** which frequencies are present and how strongly.

A single FFT can be sensitive to the selected time window. DSAS uses Welch PSD for a more stable average measurement.

### 5.13 Power spectral density (PSD)

PSD describes how signal power is distributed across frequency. It is useful for distinguishing patterns such as:

- A narrow carrier peak.
- Broad noise across many frequencies.
- A pulse train whose spectral shape changes with time.

`scanner.calculate_psd_metrics()` calls `scipy.signal.welch()` with a 256-sample segment and returns:

- Frequency bins.
- Linear PSD values.
- PSD values in dB.
- Mean channel power.
- Peak PSD in dB.

The scanner's priority formula uses mean channel power. The returned PSD arrays are available for future detailed spectrum views.

### 5.14 Welch's method

Welch's method divides a long signal into shorter segments, calculates a spectrum for each segment, and averages the results. Averaging reduces some of the random variation of a single FFT.

The current `nperseg=256` setting means each PSD segment contains 256 IQ samples. It is a practical compromise for this short prototype record.

### 5.15 STFT and spectrogram

The short-time Fourier transform, or STFT, applies a frequency analysis to many small windows as the signal progresses through time. A **spectrogram** displays the result as an image:

- Horizontal axis: time.
- Vertical axis: frequency.
- Color: measured power.

The dashboard uses this view to make radar bursts and other time-varying behavior easier to see than in a single full-record spectrum.

### 5.16 Amplitude, phase, and instantaneous frequency

The classifier uses amplitude and phase statistics because different signal types have different behaviors.

- **Amplitude:** `abs(IQ)` for each sample.
- **Phase:** `angle(IQ)` for each sample.
- **Unwrapped phase:** phase with artificial jumps at `+pi` and `-pi` removed.
- **Instantaneous frequency:** the change in unwrapped phase from one sample to the next, scaled by the sample rate.

A stable carrier tends to have a more regular phase progression. Noise produces irregular phase changes. Pulses produce strong changes between active and inactive regions.

### 5.17 PAPR

Peak-to-average power ratio, or PAPR, compares the largest power with the average power:

```text
PAPR_dB = 10 * log10(peak_power / mean_power)
```

A signal that is usually quiet but occasionally has a strong pulse can have a high PAPR. This makes PAPR useful for detecting pulse-like radar behavior.

### 5.18 Standard deviation

Standard deviation measures how much values vary around their average. A low amplitude standard deviation means the amplitude is relatively steady. A high value means it changes more.

### 5.19 Kurtosis

Kurtosis measures how heavy the tails or peaks of a distribution are compared with a normal-shaped distribution. In this project it is an additional indicator of unusual amplitude peaks. It is not a complete radar detector by itself.

### 5.20 Spectral flatness

Spectral flatness compares the geometric mean of PSD values with their arithmetic mean:

```text
flatness = geometric_mean(PSD) / arithmetic_mean(PSD)
```

Interpretation:

- A value closer to 1 suggests energy is spread more evenly, like broadband noise.
- A lower value suggests energy is concentrated in a smaller number of frequency bins, like a narrow carrier.

This is an indicator, not a guarantee. Windowing, sample length, and noise change the result.

### 5.21 Spectral centroid and spread

The spectral centroid is the power-weighted center of the frequency distribution. It is similar to a balance point: frequencies with more energy pull the center toward themselves.

Spectral spread measures how far the energy is distributed around that center. A small spread suggests concentrated energy. A larger spread suggests wider occupied bandwidth.

### 5.22 Random Forest

A Random Forest is a collection of decision trees. Each tree asks simple questions about the input features, such as whether power is above a range or PAPR is high. The forest combines the votes from many trees.

Advantages for this MVP:

- It works well with small engineered feature vectors.
- It trains quickly on a CPU.
- It does not need a GPU or a deep-learning framework.
- It can provide class probabilities used as a confidence-like value.

The current model uses 100 trees and maximum depth 10.

### 5.23 StandardScaler

The nine features have different units and scales. For example, power is in dB, frequency is in Hz, flatness is near 0 to 1, and spread is in Hz. `StandardScaler` transforms each feature so that the training set has approximately zero mean and unit standard deviation.

The scaler is fitted on training data and then reused during prediction. It must not be refitted separately for every new channel.

### 5.24 Confidence

The classifier returns the highest class probability from the Random Forest as a percentage. In this prototype it is best described as **model confidence**.

It is not a calibrated probability that a real emitter is hostile. Calibration and field validation would require representative, labeled, independent data.

---

## 6. The four simulated signal classes

The simulator creates four classes so the complete analysis pipeline has different patterns to compare.

### 6.1 Background Noise

Background noise is complex AWGN with a low power setting. It should have:

- Low average power.
- Broad, random spectral energy.
- Irregular phase.
- No intentional pulse structure.

In a default batch it is generated near `-22 dB`. The exact measured value varies because the record is finite and random.

### 6.2 Civilian Broadcast

The civilian signal is a continuous phase-modulated carrier with AWGN added to it. It is not a full Wi-Fi or FM implementation; it is a simple mathematical stand-in for an ordinary continuous communication.

Current model:

- Carrier amplitude: `0.8`.
- Default carrier frequency: `50 kHz`.
- Low-frequency message: `1 kHz` sinusoid.
- Phase modulation index: `1.5`.
- Default SNR: `15 dB`.
- Batch carrier range: approximately `20 kHz` to `100 kHz`.

Expected visual behavior:

- A relatively continuous waveform.
- A narrow or structured frequency concentration around the carrier and its modulation components.
- Lower PAPR than a pulsed radar signal.

### 6.3 Hostile Radar

The radar model is a high-amplitude carrier that switches on for a short pulse and then switches off until the next pulse repetition interval (PRI).

Current default parameters:

- Pulse width: `20 microseconds`.
- PRI: `100 microseconds`.
- Carrier frequency: `200 kHz` by default.
- Pulse amplitude: `3.5`.
- Baseline noise: approximately `-18 dB`.

At a 1 MHz sample rate, 20 microseconds is about 20 samples and 100 microseconds is about 100 samples. Over 1,024 samples, the record contains roughly ten pulse periods. The batch generator varies pulse width between 10 and 30 microseconds and carrier frequency between 150 and 300 kHz.

Expected visual behavior:

- High peaks separated by quieter gaps.
- High PAPR.
- Repeating vertical structures in a spectrogram.
- A strong, structured spectral signature.

The word "hostile" is a simulation label. The waveform alone does not prove intent.

### 6.4 Hostile Jammer

The jammer is high-power broadband complex Gaussian noise. It has no narrow carrier; it raises energy across much of the channel.

Current parameters:

- Default jammer power: `+10 dB`.
- Batch power range: approximately `+5 dB` to `+15 dB`.

Expected visual behavior:

- High mean power.
- Broad spectral occupancy.
- High spectral flatness relative to a narrow carrier.
- A noise-like waveform with no regular pulses.

Again, the class is a simulated label for testing the pipeline, not a real-world attribution.

### 6.5 Comparison table

| Class | Main distinguishing idea | Typical priority behavior |
| --- | --- | --- |
| Background Noise | Low-power random signal | Usually zero or low priority |
| Civilian Broadcast | Continuous structured carrier | Usually low severity, energy may still exceed threshold |
| Hostile Radar | Short repeating high-amplitude pulses | HIGH multiplier |
| Hostile Jammer | High-power broadband random signal | CRITICAL multiplier |

---

## 7. What a batch is

A **batch** is one snapshot containing all monitored channels. It is not a machine-learning batch in this context; it is a collection of channel records used for one scan.

The function `generate_spectrum_batch(num_channels=10)` returns a dictionary. The key is the channel ID, and the value is a record containing the data and metadata.

### 7.1 Channel record shape

```python
{
    "iq": np.ndarray,                 # complex128, shape (1024,)
    "type": "Hostile Radar",         # simulated class label
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

### 7.2 Ground truth versus prediction

The simulator knows the class because it created the signal. That known label is called **ground truth**. It is useful for testing whether the classifier is correct.

The classifier does not normally receive the ground-truth label. It sees the IQ samples and produces a prediction. Keeping these concepts separate prevents a future real-input adapter from accidentally giving the answer to the model.

### 7.3 Why the default batch covers all classes

For a useful demonstration, a batch with at least four channels includes every class at least once. Any additional channels are selected using the configured probability distribution. This makes the demo predictable while keeping the larger batch varied.

---

## 8. Code architecture and mechanisms

The project is split into four Python modules.

```text
simulator.py -> scanner.py -> classifier.py
       \             |             /
                app.py
```

The actual relationship is that the scanner calls the classifier for each channel, and the dashboard calls the simulator and scanner. The arrow diagram represents the data flow rather than a strict import order.

### 8.1 `simulator.py`: synthetic input engine

This module creates all IQ data. It does not depend on Streamlit or scikit-learn.

#### `generate_awgn()`

Purpose: create complex Gaussian noise.

Important parameters:

- `num_samples`: number of complex samples.
- `noise_power_db`: requested average complex power.
- `rng`: optional random-number generator for reproducible tests.

The function creates I and Q components with equal standard deviation and returns a `complex128` NumPy array.

#### `generate_civilian_signal()`

Purpose: create a continuous phase-modulated signal plus noise.

Mechanism:

1. Create a time array from the sample rate.
2. Create a 1 kHz message tone.
3. Add the message to the carrier phase.
4. Convert the phase to a complex exponential.
5. Add AWGN at the requested SNR.

#### `generate_hostile_radar()`

Purpose: create a pulsed carrier.

Mechanism:

1. Convert pulse width and PRI from seconds to samples.
2. Create a Boolean pulse mask using modulo arithmetic.
3. Multiply a complex carrier by that mask.
4. Add low-level baseline noise.

#### `generate_hostile_jammer()`

Purpose: create high-power broadband complex noise. It is a convenience wrapper around `generate_awgn()` with a higher power setting.

#### `generate_spectrum_batch()`

Purpose: create the complete channel dictionary. It assigns classes, generates IQ, calculates center frequencies, and attaches metadata.

#### Validation and reproducibility

The simulator rejects invalid sample counts, sample rates, frequencies, and pulse settings. An optional random generator makes it possible to create repeatable fixtures for automated tests.

### 8.2 `scanner.py`: measurement and prioritization engine

This module turns raw IQ records into a ranked pandas DataFrame.

#### `calculate_psd_metrics()`

Purpose: calculate frequency-domain metrics for one IQ array.

Mechanism:

1. Validate that IQ values are finite and there are at least two samples.
2. Call `scipy.signal.welch()` with a 256-sample segment.
3. Shift the frequency array so negative frequencies appear in the expected order.
4. Calculate mean complex power from `mean(abs(iq)**2)`.
5. Convert linear PSD and mean power into dB values.
6. Return frequencies, PSD, PSD dB, mean power, peak PSD, and the segment length.

#### `scan_and_prioritize()`

Purpose: scan every record, classify it, and sort it.

For each channel:

1. Read `iq` and `sample_rate`.
2. Measure PSD and mean power.
3. Ask the classifier for a predicted class and threat level.
4. Calculate power excess:

   ```text
   power_excess = max(0, power_db - threshold_db)
   ```

5. Select a threat multiplier:

   | Threat level | Multiplier |
   | --- | ---: |
   | CRITICAL | 2.5 |
   | HIGH | 1.8 |
   | MEDIUM | 1.2 |
   | LOW | 1.0 |

6. Calculate:

   ```text
   priority_score = power_excess * threat_multiplier
   ```

7. Add the result to a record and finally sort all records in descending priority order.

#### Why use both energy and threat level?

Energy alone would put every loud signal at the top, including ordinary strong communications. Classification alone could prioritize a weak signal that is below the measured noise floor. Combining them makes the ranking more useful and remains easy to explain.

#### Fallback behavior

If scikit-learn is unavailable, the scanner can use a heuristic fallback. In a real deployment, fallback mode should be visible to the operator because it is not equivalent to the trained model.

### 8.3 `classifier.py`: feature and ML engine

This module contains the only main class in the current backend: `RFSignalClassifier`.

#### `extract_signal_features()`

Purpose: reduce 1,024 complex samples to nine numerical values.

The exact feature order is:

1. Mean power in dB.
2. PAPR in dB.
3. Amplitude standard deviation.
4. Amplitude kurtosis.
5. Phase standard deviation.
6. Instantaneous-frequency standard deviation.
7. Spectral flatness.
8. Spectral centroid.
9. Spectral spread.

The order must remain unchanged between training and prediction. Every feature is checked or converted so that normal supported inputs produce finite floating-point values.

#### Synthetic training data

`_generate_synthetic_training_data()` creates samples for each of the four classes, extracts the same nine features, and returns:

- `X`: a numeric feature matrix.
- `y`: a label array.

The public default is 300 examples per class, or 1,200 training rows total. The data stays in memory.

#### `RFSignalClassifier` class

The class owns two fitted objects:

- `StandardScaler` for feature normalization.
- `RandomForestClassifier` for prediction.

Methods:

- `__init__()`: creates the scaler and 100-tree model.
- `train(samples_per_class=300)`: generates data, fits the scaler, and trains the forest.
- `save(filepath)`: explicitly writes the trained model and scaler to a local pickle file.
- `load(filepath)`: loads a previously saved model if it exists.
- `predict(iq_samples, sample_rate)`: extracts features, scales them, predicts a class, and maps the result to severity and status.

Normal dashboard use trains a process-local model on first use. It does not require a committed model artifact.

#### Class-to-severity mapping

```text
Hostile Jammer      -> CRITICAL / BROADBAND JAMMING
Hostile Radar       -> HIGH     / PULSED RADAR LOCK
Civilian Broadcast  -> LOW      / STANDARD COMM
Background Noise    -> LOW      / CLEAR
```

#### Shared classifier instance

The module exposes `classify_channel()` as a convenience function backed by one shared classifier instance. This avoids retraining for every channel in the same Python process.

### 8.4 `app.py`: analyst dashboard

The Streamlit app is the user-facing layer. It should not duplicate the signal-processing formulas; it uses the scanner and classifier outputs.

#### Sidebar controls

- **Noise Floor Threshold:** from `-25 dB` to `+10 dB`.
- **Enable Auto-Refresh:** periodically creates a new batch when supported by the installed Streamlit version.
- **Auto-refresh interval:** controls the refresh interval.
- **Manual Sweep:** generates a new batch and re-runs the pipeline.
- **Clear Threat Log:** clears current-session incidents.

#### KPI cards

- Monitored Channels.
- Signals Above Threshold.
- Critical Threats.
- High Threats.

#### Live Spectrum Monitor

The bar chart shows aggregate channel power against center frequency. The red line shows the selected threshold. The waterfall stores a bounded history of aggregate power rows.

Important distinction: this is currently a channel-power monitor, not a single continuous wideband FFT plot. The scanner already returns PSD arrays that can support a future true frequency-bin view.

#### Prioritized Threat Feed

The feed displays the scanner DataFrame in priority order. Rows are styled by threat level. HIGH and CRITICAL rows are copied into `st.session_state.threat_log`, which survives Streamlit reruns but not a process restart.

#### Signal Inspector

The user selects a channel and sees:

- The real part of IQ as I.
- The imaginary part of IQ as Q.
- A two-sided STFT spectrogram.

The waveform shows sample values over time. The spectrogram shows where energy appears in frequency and when it appears.

#### Streamlit session state

Session state keeps the current batch, current analysis, incident history, scan number, and bounded waterfall history. It is temporary application state, not durable storage.

---

## 9. Scanner result and classifier result contracts

Stable contracts let multiple contributors work safely.

### 9.1 Classifier result

```python
{
    "classification": "Hostile Jammer",
    "confidence": 98.2,
    "threat_level": "CRITICAL",
    "status": "BROADBAND JAMMING",
    "power_db": 10.1,
    "papr_db": 8.0,
    "features": np.ndarray,  # shape (9,)
}
```

### 9.2 Scanner DataFrame

The DataFrame contains:

| Column | Meaning |
| --- | --- |
| `Channel` | Channel ID |
| `Center Freq (MHz)` | Center frequency metadata |
| `Power (dB)` | Mean IQ power in dB |
| `Classification` | Model or fallback class |
| `Confidence (%)` | Model confidence-like percentage |
| `Threat Level` | LOW, MEDIUM, HIGH, or CRITICAL |
| `Status` | Human-readable status string |
| `Power Excess (dB)` | Power above threshold, never negative |
| `Threat Multiplier` | Severity multiplier used in the score |
| `Priority Score` | Final descending-rank score |
| `Timestamp` | Time associated with the scan |

---

## 10. A complete example calculation

Suppose one channel produces:

```text
measured power = -3 dB
threshold       = -10 dB
classification  = Hostile Radar
threat level    = HIGH
```

Then:

```text
power_excess = max(0, -3 - (-10))
              = 7 dB

priority_score = 7 * 1.8
               = 12.6
```

Suppose another channel has `+4 dB` power but is classified as a civilian broadcast:

```text
power_excess = 4 - (-10) = 14 dB
priority_score = 14 * 1.0 = 14.0
```

The civilian channel could rank above the radar in this example because it has much more excess energy. This is an important design choice: threat multipliers influence energy ranking; they do not replace the energy measurement. A future product may use a different policy, such as always placing CRITICAL alerts first, but that would be a documented rule change.

---

## 11. How to run the project

From the project root:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal, normally `http://localhost:8501`.

### Backend-only checks

```powershell
python simulator.py
python scanner.py
python classifier.py
python -m py_compile simulator.py scanner.py classifier.py app.py
```

### What to try in the dashboard

1. Read the sidebar status and confirm the Random Forest is online.
2. Look at the four KPI cards.
3. Change the noise-floor threshold and watch power excess and priority scores change.
4. Click Manual Sweep and observe the batch, feed, and plots update.
5. Open the threat feed and compare classification, confidence, power, and priority.
6. Select a radar or jammer channel in the inspector.
7. Compare the I/Q waveform with the STFT image.
8. Clear the threat log.
9. Run another sweep and observe that new HIGH/CRITICAL rows can be logged.

---

## 12. Current implementation status

| Area | Status today | Meaning |
| --- | --- | --- |
| Synthetic IQ generation | Implemented | Four mathematical signal families are available. |
| Ten-channel batch | Implemented | Default batch has documented metadata and class coverage. |
| Welch PSD | Implemented | 256-sample segment and mean-power metrics are available. |
| Priority ranking | Implemented | Threshold and threat multipliers are applied and sorted. |
| Nine-feature extraction | Implemented | Feature order is fixed and finite for supported signals. |
| Random Forest | Implemented | StandardScaler and 100-tree model train in memory. |
| Streamlit views | Implemented | Spectrum, feed, inspector, and incident log are present. |
| Automated regression suite | Next work | The repository should add committed unit tests. |
| Held-out evaluation | Next work | Training accuracy alone is not enough. |
| Durable incidents | Not in MVP | Current state is session-only. |
| Hardware input | Explicitly deferred | Requires a separate design and validation effort. |

---

## 13. Recommended next changes

These are improvements that fit the existing architecture. They should be implemented in the following order.

### Priority 0: trust and correctness

#### 13.1 Add automated backend tests

Test signal shape, complex dtype, pulse timing, power trends, PSD segment length, feature shape, finite values, priority multipliers, sorting, and empty input behavior.

#### 13.2 Add held-out classifier metrics

Create an evaluation dataset that is not used for training. Report accuracy, precision, recall, F1, confusion matrix, and per-class support. This shows whether the model generalizes to new synthetic examples.

#### 13.3 Add deterministic fixtures

Use a fixed random seed for one known example of every class. Reproducible fixtures make debugging and UI tests much easier.

#### 13.4 Explain each score

Show the formula inputs in the UI: power, threshold, power excess, threat multiplier, and final score. A beginner should be able to reproduce a row's score with a calculator.

#### 13.5 Harden error states

Handle empty batches, missing IQ, short arrays, invalid settings, and classifier failures with visible messages rather than opaque tracebacks.

### Priority 1: analyst usefulness

#### 13.6 Add incident export

Provide CSV and JSON downloads for the incident log and current scan. Keep raw IQ out of the default incident export so files remain small.

#### 13.7 Add a true frequency-bin spectrum view

Use the PSD frequency and power arrays already calculated by the scanner. Keep the current channel-level bar chart and label the two views clearly.

#### 13.8 Improve the waterfall

Store a bounded history of PSD rows with timestamps instead of repeating aggregate channel power. This makes the waterfall a real time-frequency history.

#### 13.9 Add analyst filters

Filter by threat level, class, minimum confidence, and minimum priority while keeping priority sorting as the default.

#### 13.10 Add threshold calibration

Estimate a baseline from background-only simulated sweeps, explain how it was estimated, and keep the manual slider as an override.

### Priority 2: production-shaped architecture without hardware

#### 13.11 Add an input adapter interface

Define a `read_batch()` interface. Keep the simulator as the first adapter and later support file replay. Do not couple the dashboard directly to a future SDR library.

#### 13.12 Add optional persistence

Define a repository interface with the current in-memory implementation first. A later SQLite implementation could preserve incident history across restarts.

#### 13.13 Add configuration and observability

Centralize defaults, record scan duration, model version, training time, and fallback status, and add structured logs.

#### 13.14 Add CI

Run dependency installation, tests, Python compilation, link checks, and a headless Streamlit smoke check automatically.

### Do not mix into the MVP hardening pass

Do not add live SDR hardware, real-world hostile labels, deep learning, distributed sensors, or operational control until the synthetic pipeline has tests, metrics, clear limitations, and a separate safety review.

---

## 14. Verification and acceptance criteria

### Backend acceptance

- Every supported generator returns a finite one-dimensional complex array.
- The default array shape is `(1024,)` at a 1 MHz sample rate.
- Radar pulse settings map to approximately 20 samples ON per 100 samples at the default rate.
- Welch PSD uses a 256-sample segment for the default input.
- The feature extractor returns exactly nine finite values.
- The Random Forest uses 100 estimators and maximum depth 10.
- Training with 300 examples per class meets the agreed time budget.
- All four classes appear in a default ten-channel batch.
- Priority results are sorted in descending score.
- Empty batches and invalid inputs have documented behavior.

### Dashboard acceptance

- The app loads without uncaught exceptions.
- The sidebar contains threshold, manual sweep, auto-refresh, and clear-log controls.
- Four KPI cards appear.
- All three main views are present.
- Threshold changes re-score the current batch.
- Manual sweep changes the current batch and analysis.
- The incident log is session-persistent and bounded.
- The inspector renders I/Q and STFT views.
- Synthetic-only limitations are visible in the documentation and should be visible in the UI.

### Interpretation acceptance

- A simulated class is not presented as confirmed real-world intent.
- Model confidence is not presented as a calibrated probability.
- Ground truth is used for tests, not leaked into normal model prediction.
- Hardware-free operation remains possible.

---

## 15. Five-account work division

If five Opus accounts are available, use separate branches or worktrees and keep file ownership clear.

| Account | Responsibility | Main files | Expected result |
| --- | --- | --- | --- |
| Opus-1 | Technical lead and contracts | `docs/` | Freeze interfaces, units, labels, and integration checklist. |
| Opus-2 | Simulation and DSP | `simulator.py`, `scanner.py`, backend tests | Deterministic fixtures, pulse/PSD tests, optional PSD view data. |
| Opus-3 | ML and evaluation | `classifier.py`, evaluation code, classifier tests | Held-out metrics, benchmark, stable prediction contract. |
| Opus-4 | Dashboard | `app.py`, AppTest coverage | Explainable feed, export/filter/error states, tested interactions. |
| Opus-5 | QA and release | `tests/`, verification docs | Merge, run full checks, clean-install proof, release recommendation. |

### Recommended order

1. Opus-1 freezes the data contracts.
2. Opus-2, Opus-3, and Opus-4 work in parallel within their owned files.
3. Opus-5 integrates and tests each branch.
4. Opus-1 updates this abstract if any accepted contract changes occur.

### Handoff information every account must provide

```text
Role:
Files changed:
Contract changes: none / list them
Behavior added:
Tests run and exact command:
Observed timing:
Known limitations:
Follow-up needed:
```

### Rules for parallel work

- Do not silently change feature order, labels, units, or DataFrame columns.
- Do not edit another account's owned files without agreement.
- Do not commit `__pycache__`, temporary models, or generated reports.
- Do not add dependencies without documenting the reason.
- Keep commits small and focused.
- Treat a failing test as an integration issue to investigate, not as a reason to hide the test.

---

## 16. Beginner glossary

| Term | Simple meaning |
| --- | --- |
| Amplitude | How strong a complex sample is. |
| AWGN | Mathematical random background noise. |
| Batch | One collection of all channel records in a scan. |
| Carrier | A repeating oscillation used as the base of a transmission. |
| Channel | One monitored frequency region. |
| Complex number | A number with real and imaginary parts; here it stores I and Q. |
| Confidence | The model's preference for its predicted class, not a guaranteed probability. |
| dB | Logarithmic representation of a power ratio. |
| FFT | Fast conversion from time-domain samples to frequency components. |
| Frequency bin | One discrete frequency location in an FFT or PSD array. |
| Ground truth | The known simulated label used to evaluate a prediction. |
| IQ | In-phase and quadrature representation of a signal. |
| Jammer | In this simulation, high-power broadband noise. |
| Kurtosis | A statistic related to unusually heavy tails or peaks. |
| Noise floor | Background level used as a comparison threshold. |
| Nyquist frequency | Approximately half the sample rate. |
| PAPR | Peak-to-average power ratio; useful for pulse-like behavior. |
| PSD | Power distribution across frequency. |
| Radar pulse train | Repeated high-amplitude pulses separated by quiet intervals. |
| Random Forest | Many decision trees voting on a class. |
| Sample rate | Number of numerical measurements taken per second. |
| SNR | Signal power compared with noise power. |
| Spectral centroid | Power-weighted center of frequency. |
| Spectral flatness | Indicator of whether energy is spread or concentrated across frequency. |
| Spectral spread | Width of the frequency distribution around its centroid. |
| STFT | Repeated short FFT windows that show frequency versus time. |
| Threat multiplier | Number that increases priority for a more severe class. |
| Welch method | Averaged PSD method using multiple signal segments. |
| Waterfall | A sequence of spectrum rows displayed as a heatmap over time. |

---

## 17. Final interpretation

DSAS is best understood as a transparent teaching and prototyping pipeline:

```text
IQ samples
   -> measurable signal properties
   -> model prediction
   -> explainable priority score
   -> analyst visualization
```

Its strongest value is that every result can be traced back to synthetic data, mathematical measurements, engineered features, and a simple scoring rule. Its most important limitation is that all signal classes and model behavior are synthetic. The responsible next step is therefore better testing, held-out evaluation, explainability, and replayable data—not an unsupported claim that the prototype is already a field-ready RF defense system.
