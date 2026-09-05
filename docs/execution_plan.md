# DSAS Execution Plan

## 1. Outcome to deliver

The next implementation pass should produce a reliable, demo-ready prototype that a new contributor can install, run, understand, and verify. The target is not a field-ready RF system. The target is a transparent pipeline with repeatable synthetic data, measurable classifier behavior, a usable analyst dashboard, and a clean handoff between five parallel contributors.

## 2. Work already completed

The baseline four-module pipeline is present:

- `simulator.py` generates the four required signal families.
- `scanner.py` calculates Welch PSD and ranks channels.
- `classifier.py` extracts nine features and trains a Random Forest.
- `app.py` renders the three dashboard views and session log.
- `requirements.txt` defines the runtime dependencies.

Do not spend the next pass rewriting this baseline unless a contract or failing test requires it.

## 3. Recommended 4.5-hour hardening schedule

| Time | Workstream | Output | Exit gate |
| --- | --- | --- | --- |
| 0:00-0:20 | Baseline and contracts | Run the app, record Python/package versions, freeze channel/row/result schemas. | All contributors agree on interfaces. |
| 0:20-0:55 | Tests and deterministic fixtures | Add tests for signal shape, PSD window, feature length, score formula, and empty input. | Backend tests pass twice in a row. |
| 0:55-1:35 | Simulator and DSP hardening | Add seeded fixtures, exact pulse-duty checks, and optional true frequency-bin spectrum data. | Signal properties are observable and finite. |
| 1:35-2:20 | Classifier evaluation | Add held-out synthetic data, confusion matrix, per-class metrics, and a saved evaluation report. | Training meets the time budget and every class is evaluated. |
| 2:20-3:20 | Dashboard improvements | Add explainability panel, better empty/error states, export action, and replace deprecated Streamlit parameters when compatible. | AppTest has no exceptions and all user actions work. |
| 3:20-3:55 | Integration | Connect the new scanner/classifier fields, confirm threshold behavior, and confirm incident-log deduplication. | One manual sweep updates every view consistently. |
| 3:55-4:20 | Documentation and demo path | Update screenshots/notes if available, add runbook, and write the two-minute demo script. | A new person can run the project using docs only. |
| 4:20-4:30 | Release gate | Run all tests, `git diff --check`, and a clean install smoke test. | No untracked generated artifacts or known unexplained failures. |

## 4. Dependency order

```text
contracts/tests
    ├── simulator + scanner hardening
    ├── classifier evaluation
    └── dashboard improvements
              \       |       /
                integration
                    |
              verification + demo
```

The simulator and classifier can be developed in parallel after the feature/record contracts are frozen. Dashboard work can use deterministic fixtures and does not need to wait for a better model.

## 5. Definition of done

The next pass is complete only when all of the following are true:

- A clean environment installs from `requirements.txt`.
- The default run creates ten channels of complex IQ with the documented sample rate and length.
- The scanner uses a 256-sample Welch segment and the priority formula is tested directly.
- The feature extractor returns nine finite values for every supported signal class.
- The model reports per-class held-out metrics and remains within the training-time budget.
- Manual sweep, threshold change, auto-refresh, signal selection, and clear-log actions are exercised.
- The UI explains that all data is synthetic and that confidence is model confidence, not a field-grade probability.
- The incident log behavior is documented as session-only or upgraded to an explicit persistence mechanism.
- The repository has no generated model, cache, or temporary files committed accidentally.

## 6. Demo script for a new operator

1. Install dependencies and start Streamlit.
2. Point out that the system is synthetic and hardware-free.
3. Show the four KPI values and explain that the current batch includes all signal families.
4. Move the threshold slider and show priority scores change without changing IQ data.
5. Click Manual Sweep and show the channel order, power chart, and threat feed update together.
6. Open a HIGH or CRITICAL channel in the inspector and explain the waveform/STFT difference.
7. Open the incident log, clear it, then run another sweep to show the lifecycle.
8. Close with the limitation: synthetic data demonstrates the pipeline but does not validate real-world detection.

## 7. If time runs short

Keep P0 work from [feature_backlog.md](feature_backlog.md): tests, evaluation metrics, dashboard error handling, and reproducible demo behavior. Defer durable storage, authentication, hardware adapters, and cosmetic redesign.
