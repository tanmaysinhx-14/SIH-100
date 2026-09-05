# Copy-Paste Prototype Prompts for the Five Opus Accounts

Use the shared context first, then give one role prompt to each account. These prompts are intentionally bounded so parallel agents improve the same prototype instead of creating five incompatible rewrites.

## Shared context for every account

```text
You are contributing to the DSAS repository, a defensive, synthetic-only RF spectrum-awareness prototype.

Read docs/README.md, docs/current_scope.md, docs/execution_plan.md, and docs/opus_team_plan.md before editing. Inspect the current files and git status. Preserve existing user work. Do not add SDR hardware, network capture, deep learning, GPU dependencies, or unrelated product features.

The current default contract is: ten channels, 1,024 complex IQ samples per channel, 1 MHz sample rate, four signal labels (Background Noise, Civilian Broadcast, Hostile Radar, Hostile Jammer), nine classifier features in the documented order, Welch PSD with a 256-sample segment, and priority = max(0, power_db - threshold_db) * threat_multiplier.

Work only in your assigned files. If you need a contract change, document it explicitly and stop before changing another owner's files. Add focused tests or executable verification for every behavior you change. Do not commit generated caches, models, or temporary artifacts.
```

## Opus-1: technical lead and contract owner

```text
Own the documentation and interface contract for the next DSAS hardening pass.

Tasks:
1. Verify the current source against docs/current_scope.md.
2. Freeze the Channel Record, classifier result, scanner DataFrame, labels, units, and error behavior.
3. Update docs/README.md, docs/current_scope.md, docs/architecture.md, docs/workflow.md, and docs/execution_plan.md when evidence requires it.
4. Produce a short integration checklist for Opus-2 through Opus-5.
5. Do not implement UI or model features in this branch.

Deliverable: documentation-only contract commit plus a handoff using the template in docs/opus_team_plan.md.
Verification: inspect all four modules, run python -m py_compile simulator.py scanner.py classifier.py app.py, and report any contract mismatch.
```

## Opus-2: simulator and DSP engineer

```text
Own simulator.py and scanner.py, plus focused tests for those modules.

Tasks:
1. Add deterministic fixtures or a seeded fixture factory for one instance of each signal class.
2. Test complex dtype, sample count, sample rate assumptions, power trends, radar pulse width/PRI, and jammer broadband behavior.
3. Test calculate_psd_metrics() uses nperseg=256 for a 1,024-sample input and returns finite aligned arrays.
4. Test the priority formula and all four threat multipliers with controlled channel records.
5. If adding true frequency-bin spectrum data, preserve existing DataFrame columns and document the new optional field.

Do not change classifier.py or app.py. Do not use ground-truth labels in the normal ML path.

Deliverable: backend hardening commit, tests, and timing/evidence report.
Verification: run the focused tests twice and run python scanner.py.
```

## Opus-3: classifier and evaluation engineer

```text
Own classifier.py and classifier tests/evaluation utilities.

Tasks:
1. Preserve the exact nine-feature order and make every feature finite for all supported signals.
2. Add a seeded held-out evaluation path separate from training data.
3. Report accuracy, precision, recall, F1, confusion matrix, and per-class support.
4. Benchmark train(samples_per_class=300) and keep it within the project budget.
5. Add compact explanation metadata if useful, but do not change existing prediction keys without documenting the contract change.
6. Verify explicit save/load still produces the same prediction; normal dashboard prediction must not require a committed model file.

Do not edit simulator.py, scanner.py, or app.py. Use the existing signal generators.

Deliverable: classifier/evaluation commit and exact benchmark output.
Verification: run the focused classifier tests and a known-signal prediction check for all four classes.
```

## Opus-4: dashboard engineer

```text
Own app.py and dashboard tests.

Tasks:
1. Preserve the sidebar controls, four KPIs, three tabs, threat styling, incident log, waveform, and STFT behavior.
2. Add the highest-value P0/P1 UX improvements that fit the time box: explainable score details, robust empty/error states, and incident export or analyst filtering.
3. Keep all UI labels explicit that data is synthetic and confidence is model confidence.
4. Test manual sweep, threshold changes, clear-log behavior, channel selection, and auto-refresh with Streamlit AppTest.
5. Avoid deprecated Streamlit calls when the minimum supported version remains compatible.

Do not change backend contracts. If an extra field is needed, mock it locally first and request the contract change from Opus-1.

Deliverable: dashboard commit plus AppTest evidence.
Verification: run the app test with zero exceptions and list any warnings separately from failures.
```

## Opus-5: QA and release engineer

```text
Own tests/, integration verification, and release evidence.

Tasks:
1. Start from the contract commit and inspect every incoming branch before merging.
2. Build a small test matrix covering simulator, scanner, classifier, and Streamlit AppTest.
3. Run tests after each merge and isolate regressions to the smallest reproduction.
4. Run a clean-install smoke check from requirements.txt, py_compile, backend commands, and a headless Streamlit health check.
5. Check git diff --check and remove only generated artifacts created by verification.
6. Update docs/verification_plan.md with actual commands and results; do not hide warnings.

Do not add product features unless needed to make a failing test pass. Do not approve unexplained contract changes.

Deliverable: final verification commit/report and a release recommendation.
```
