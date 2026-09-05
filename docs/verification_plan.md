# Verification Plan

## 1. Verification levels

| Level | Purpose | Evidence |
| --- | --- | --- |
| Unit | Prove mathematical and contract behavior. | Focused pytest tests for simulator, scanner, and classifier. |
| Integration | Prove the complete batch-to-DataFrame path. | One seeded ten-channel scan with score and sort assertions. |
| UI harness | Prove Streamlit widgets and tabs execute without exceptions. | `streamlit.testing.v1.AppTest`. |
| Runtime | Prove the server starts and is reachable. | Headless Streamlit health endpoint returns `200` and `ok`. |
| Human demo | Prove a newcomer can operate and explain it. | Runbook checklist and two-minute demo script. |

## 2. Backend test matrix

### Simulator

- Each generator returns a one-dimensional complex NumPy array.
- Default generator length is 1,024.
- AWGN power is close to its configured dB target over a sufficiently large sample.
- Civilian output is continuous and has the requested signal/noise relationship within expected random variation.
- Radar output contains approximately 20 samples ON every 100 samples at the default 1 MHz sample rate.
- Jammer output has materially higher power than background noise.
- A ten-channel batch contains the documented keys and all four classes.
- Invalid sample counts, rates, and frequencies raise `ValueError`.

### Scanner

- Welch is called with a 256-sample segment for the default 1,024-sample input.
- Frequency and PSD arrays have equal length and finite values.
- Power dB is derived from mean complex magnitude-squared power.
- A channel below threshold has zero power excess.
- Priority multipliers are exactly 2.5, 1.8, 1.2, and 1.0 for CRITICAL, HIGH, MEDIUM, and LOW.
- Results sort monotonically by descending `Priority Score`.
- An empty batch returns the documented columns and no rows.

### Classifier

- Feature vectors have shape `(9,)` and contain finite floats.
- The class set is exactly the four documented labels.
- `train(samples_per_class=300)` completes under the agreed time budget on the target machine.
- Held-out evaluation has at least one example per class.
- Prediction includes class, confidence, threat level, status, power, PAPR, and features.
- Explicit save/load preserves predictions.
- Zero/constant and very short invalid inputs behave according to the contract.

## 3. UI test matrix

Use `AppTest` for:

- Initial load: one title, three tabs, four KPI metrics, and no uncaught exceptions.
- Manual sweep: scan number and current data update.
- Threshold change: current batch is re-scored without needing a new IQ batch.
- Clear threat log: current log becomes empty and remains empty until a later scan creates a new incident.
- Auto-refresh path: the app executes without exceptions when enabled.
- Signal selector: waveform and spectrogram render for a valid channel.
- Fallback path: the app explains when the classifier is unavailable instead of crashing.

## 4. Commands

Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m py_compile simulator.py scanner.py classifier.py app.py
python simulator.py
python scanner.py
python classifier.py
pytest -q
streamlit run app.py --server.headless true --server.port 8501
```

The Streamlit health check is:

```powershell
Invoke-WebRequest http://localhost:8501/_stcore/health -UseBasicParsing
```

If `pytest` has not been added yet, record that as a missing P0 deliverable rather than claiming the test gate passed based only on manual smoke output.

## 5. Performance gates

- Classifier training with 300 samples per class: target under 2 seconds, hard limit under 5 seconds.
- A default ten-channel scan: should be fast enough for interactive manual sweeps; record measured time after any feature extraction changes.
- Streamlit first render: record it after the classifier is trained or cached.
- Bound all in-memory history buffers; no feature should grow memory without a limit.

## 6. Safety and interpretation checks

- The UI and runbook state that the data is synthetic.
- The UI does not claim that model confidence is a calibrated probability.
- Ground-truth metadata is never available to a future real-input classification path.
- No hardware or external RF collection is required to run the prototype.
- Demo labels describe simulated classes, not confirmed real-world hostile actors.

## 7. Release checklist

Before calling a milestone complete:

- [ ] Source contracts and docs agree.
- [ ] Focused backend tests pass.
- [ ] Held-out classifier metrics are recorded.
- [ ] AppTest passes with zero exceptions.
- [ ] Headless health endpoint is `200 / ok`.
- [ ] `git diff --check` is clean apart from line-ending notices.
- [ ] No generated caches, model files, or temporary reports are untracked.
- [ ] Known limitations and deferred features are in the handoff.
