# Feature Backlog and Recommended New Changes

This backlog turns the current MVP into concrete work items. Priority is based on whether a feature improves correctness and demo trust before adding operational complexity.

## P0 - build next

### F-01: Automated backend test suite

**Why:** The current system has smoke validation but no committed regression tests.

**Scope:** Add tests for generator shapes/dtypes, signal power trends, pulse duty cycle, Welch `nperseg=256`, finite nine-feature vectors, four-class training, score multiplication, sorted results, and empty batches.

**Acceptance:** `pytest` passes on a clean install; tests do not depend on a live Streamlit server or a generated model file.

### F-02: Held-out classifier evaluation

**Why:** Training accuracy on synthetic training rows does not show generalization.

**Scope:** Generate a separate seeded evaluation set, report accuracy, precision, recall, F1, confusion matrix, and per-class support. Save a human-readable report under `docs/` or `reports/` only when explicitly requested.

**Acceptance:** Every class has held-out examples; evaluation finishes within the project time budget; the dashboard or runbook explains what the metrics mean.

### F-03: Reproducible scan fixtures

**Why:** Random sweeps make UI and integration bugs hard to reproduce.

**Scope:** Add a fixed-seed fixture or a small fixture factory that creates one known noise, civilian, radar, and jammer channel.

**Acceptance:** Re-running the fixture produces the same class assignment and stable enough metrics for tests.

### F-04: Explainable threat details

**Why:** A new user needs to understand why a row is high priority.

**Scope:** Show power excess, threat multiplier, PAPR, confidence, and a short rule-based explanation in the inspector or threat feed.

**Acceptance:** Selecting a row makes the score traceable to the documented formula without reading source code.

### F-05: Robust dashboard states

**Why:** Empty batches, missing IQ, too-short IQ, and model errors should not produce an opaque traceback.

**Scope:** Add friendly error messages, disabled inspector state for empty data, and a visible fallback-mode explanation.

**Acceptance:** AppTest covers normal, empty, and classifier-unavailable paths with zero uncaught exceptions.

## P1 - high-value analyst features

### F-06: CSV/JSON incident export

Add a download button for the current incident log and scan results. Preserve a timestamp, channel, threat level, score, classification, and confidence. Do not serialize raw IQ into every incident export by default.

### F-07: True frequency-bin spectrum view

Keep the channel-level priority bar but add an optional PSD plot using the `freqs` and `psd_db` arrays from `calculate_psd_metrics()`. Clearly label aggregate channel power versus frequency-bin PSD.

### F-08: Real waterfall history

Store a bounded history of PSD rows with timestamps rather than repeating aggregate power. Add a fixed maximum history size so memory use is predictable.

### F-09: Analyst filters and sorting

Allow filtering by threat level, classification, minimum confidence, and minimum priority. Keep the default view sorted by priority.

### F-10: Threshold/noise-floor calibration

Add a calibration action that estimates a baseline from background-only synthetic sweeps, records the method, and shows how the threshold was selected. Keep the manual slider as an override.

## P2 - production-shaped, still hardware-free

### F-11: Input adapter interface

Define a protocol such as `read_batch() -> ChannelBatch` and implement the current simulator as the first adapter. This makes future file replay or SDR integration possible without coupling it to the dashboard.

### F-12: Persistent storage abstraction

Add a small repository interface with an in-memory implementation and a SQLite implementation. Keep persistence optional and do not introduce it before incident semantics are tested.

### F-13: Configuration and observability

Move defaults into a typed configuration object, add structured logs, record scan duration, and expose classifier training time and model version in the UI.

### F-14: CI quality gate

Add a CI job for install, tests, compile checks, lint/type checks if adopted, and a headless Streamlit smoke run.

## Defer deliberately

- Live SDR hardware and RF collection.
- Real-world hostile-signal labeling or attribution.
- Deep learning and GPU dependencies.
- Distributed sensor fusion.
- Authentication and multi-user permissions.

Those changes need new requirements, safety review, and a different validation plan. They should not be mixed into the short MVP hardening pass.
