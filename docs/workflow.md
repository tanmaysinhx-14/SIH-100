# DSAS Runtime Workflow

## One sweep from input to dashboard

### 1. Generate a batch

`simulator.generate_spectrum_batch()` creates ten channel records. Each record contains 1,024 complex samples at a 1 MHz sample rate. The default simulator includes background noise, a continuous civilian signal, a radar pulse train, and a broadband jammer so a demo can show all four classes.

### 2. Measure channel energy

For every channel, `scanner.calculate_psd_metrics()` calls `scipy.signal.welch()` with a 256-sample segment. The result includes a two-sided PSD, frequency bins, peak PSD in dB, and mean complex power in dB:

```text
power_db = 10 * log10(mean(abs(iq)**2))
```

The default noise-floor threshold is -10 dB. A threshold slider lets the operator change the comparison without generating new IQ data.

### 3. Classify the signal

The classifier converts each IQ array into the fixed nine-value feature vector documented in [current_scope.md](current_scope.md). The vector is standardized and passed to a 100-tree Random Forest with maximum depth 10.

The four simulated classes map to operational severity as follows:

| Class | Severity | Status |
| --- | --- | --- |
| Hostile Jammer | CRITICAL | BROADBAND JAMMING |
| Hostile Radar | HIGH | PULSED RADAR LOCK |
| Civilian Broadcast | LOW | STANDARD COMM |
| Background Noise | LOW | CLEAR |

### 4. Compute priority

The scanner combines energy and severity:

```text
power_excess = max(0, power_db - threshold_db)
priority = power_excess * threat_multiplier
```

Multipliers are CRITICAL 2.5, HIGH 1.8, MEDIUM 1.2, and LOW 1.0. This is intentionally simple enough to explain during a demonstration.

### 5. Present results

The dashboard stores the current batch and DataFrame in Streamlit session state. It then renders:

- A power-by-channel bar chart and bounded waterfall history.
- A ranked threat feed with severity styling.
- A HIGH/CRITICAL incident log for the current session.
- I/Q traces and a two-sided STFT for a selected channel.

### 6. Repeat or clear state

Manual Sweep creates a new batch. Changing the threshold re-scores the current batch. Auto-Refresh creates periodic new batches when supported by the installed Streamlit version. Clear Threat Log empties the incident history for the current session; a later scan can create new incidents.

## What the operator should not infer

- A high score is not proof that a real hostile transmitter exists.
- Model confidence is not a calibrated probability.
- The simulator's ground-truth class is not available from a real receiver.
- A channel-level power bar is not the same as a full wideband spectrum plot.

## Failure paths

| Condition | Expected behavior |
| --- | --- |
| scikit-learn unavailable | Scanner uses a heuristic fallback and the sidebar says so. |
| Empty batch | Scanner returns an empty DataFrame with documented columns; UI should show an empty-state message after the next hardening pass. |
| Invalid/short IQ | Backend raises a clear `ValueError`; UI should convert it to a user-facing message. |
| Streamlit version lacks fragments | Auto-refresh is not available; the manual sweep remains usable. |

## Development workflow

1. Freeze or update the contracts before changing a module.
2. Add a deterministic test or fixture for the behavior.
3. Implement one workstream in its owned files.
4. Run backend tests and AppTest.
5. Update the relevant documentation and handoff.
6. Merge only after the verification gates in [verification_plan.md](verification_plan.md) pass.
