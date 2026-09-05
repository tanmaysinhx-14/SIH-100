# DSAS System Architecture

## Architectural goal

Keep the signal pipeline independent from the UI so each stage can be tested, replaced, or eventually connected to a replay file or hardware adapter without rewriting the dashboard.

## Current pipeline

```text
                synthetic-only input
                        |
                        v
              +---------------------+
              | simulator.py        |
              | complex IQ batches  |
              +----------+----------+
                         |
                         v
              +---------------------+
              | scanner.py          |
              | Welch PSD + energy  |
              | priority scoring   |
              +----------+----------+
                         |
          +--------------+--------------+
          |                             |
          v                             v
 +------------------+          +------------------+
 | classifier.py    |          | scan DataFrame   |
 | nine features    |          | ranked channels  |
 | Random Forest    |          +--------+---------+
 +---------+--------+                   |
           +---------------------------+
                       |
                       v
              +---------------------+
              | app.py              |
              | Streamlit + Plotly  |
              +---------------------+
```

## Module responsibilities

### `simulator.py`

Owns mathematical signal generation. It produces complex `I + jQ` arrays and channel metadata. It must not import Streamlit, pandas, scikit-learn, or hardware drivers.

### `scanner.py`

Owns signal energy measurement and the ranking decision. It calls the classifier through a small function boundary and returns a pandas DataFrame suitable for both tests and the dashboard.

### `classifier.py`

Owns feature extraction, synthetic training, model persistence, and prediction-to-severity mapping. It must keep the nine-feature order stable.

### `app.py`

Owns controls, session state, charts, incident history, and user-facing error messages. It should consume scanner/classifier outputs instead of duplicating DSP or classification logic.

## Interface contracts

### Input contract

`generate_spectrum_batch()` returns `dict[channel_id, channel_record]`. A channel record has:

- `iq`: finite complex NumPy array, 1,024 samples by default.
- `sample_rate`: positive float, 1 MHz by default.
- `freq_mhz` and `center_freq_mhz`: channel center frequency.
- `type`, `signal_class`, and `ground_truth`: synthetic metadata.

Only `iq` and `sample_rate` are signal-processing inputs. A future real-input adapter should omit `ground_truth`.

### Classifier contract

`classify_channel(iq, sample_rate)` returns a dictionary with classification, confidence, threat level, status, power, PAPR, and the nine-value feature vector.

### Scan contract

`scan_and_prioritize(channels, threshold_db)` returns the columns documented in [current_scope.md](current_scope.md). The priority score is calculated only from measured power and the classifier's threat level.

## State ownership

| State | Owner | Lifetime |
| --- | --- | --- |
| Current IQ batch | `st.session_state.current_channels` | Current Streamlit session |
| Current scan DataFrame | `st.session_state.current_analysis` | Current Streamlit session |
| Threat incident log | `st.session_state.threat_log` | Current Streamlit session, maximum 50 rows |
| Waterfall power history | `st.session_state.waterfall_history` | Current Streamlit session, maximum 12 rows |
| Trained classifier | Process-local singleton | Current Python process |

## Extension points

The next clean extension is an input adapter:

```python
class ChannelSource(Protocol):
    def read_batch(self) -> dict[int, dict[str, Any]]: ...
```

The simulator would implement this interface first. A file-replay adapter could be added next. Hardware integration is intentionally outside the current project scope and requires a separate design and validation review.

## Design constraints

- No SDR drivers, network capture, or external RF input.
- No deep learning or GPU dependency.
- Training and scanning must remain interactive.
- Every user-visible score should be traceable to a documented measurement or rule.
- Synthetic ground truth is useful for testing but must never be confused with a real-world label.
