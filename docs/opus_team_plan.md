# Five-Account Opus Work Split

This plan assumes five independent Opus accounts contribute to the same repository. The accounts should work in separate branches or worktrees and merge in the order below. If the accounts instead edit one shared checkout, enforce the file ownership table strictly and do not have two accounts modify the same file at once.

## Team map

| Account | Role | Primary ownership | Main deliverable |
| --- | --- | --- | --- |
| Opus-1 | Technical lead and contract owner | `docs/`, public interfaces, integration checklist | Frozen contracts, execution coordination, final docs |
| Opus-2 | Simulation and DSP engineer | `simulator.py`, `scanner.py`, backend tests for those modules | Deterministic fixtures, DSP checks, optional frequency-bin PSD |
| Opus-3 | ML and evaluation engineer | `classifier.py`, evaluation utilities, classifier tests | Held-out metrics, model explanation fields, training benchmark |
| Opus-4 | Dashboard engineer | `app.py`, dashboard/AppTest coverage | Analyst UX, export/filter/error states, updated visual behavior |
| Opus-5 | QA, integration, and release engineer | `tests/`, `requirements.txt` only when needed, release/runbook checks | Full verification, conflict resolution, clean-install proof |

## Why this split works

- Opus-1 prevents the simulator, classifier, and UI agents from inventing incompatible data shapes.
- Opus-2 owns mathematical signal behavior and does not need to wait for UI work.
- Opus-3 can train and evaluate against fixed fixtures without editing the dashboard.
- Opus-4 can build UI features against the existing scanner contract and mock data.
- Opus-5 integrates after the other work is stable and is accountable for evidence rather than feature volume.

## Integration order

### Step 1 - Opus-1 freezes contracts

Before parallel implementation, record:

- Channel record keys and dtypes.
- The exact nine-feature order.
- Threat labels and severity mapping.
- Scanner DataFrame columns.
- Error behavior for empty, short, and invalid IQ arrays.

Commit this contract first. Other accounts rebase from it.

### Step 2 - Opus-2, Opus-3, and Opus-4 work in parallel

Each account must use deterministic fixtures and must not change another account's owned files. UI changes should treat backend outputs as an interface, not reach into classifier internals.

### Step 3 - Opus-5 integrates and tests

Opus-5 merges one branch at a time, runs backend tests after each merge, then runs AppTest and the clean-install check. Any contract change goes back to Opus-1 for documentation before it is accepted.

### Step 4 - Opus-1 publishes the demo handoff

The final handoff must include the run command, known limitations, test evidence, and a list of intentionally deferred features.

## Suggested four-and-a-half-hour parallel schedule

| Time | Opus-1 | Opus-2 | Opus-3 | Opus-4 | Opus-5 |
| --- | --- | --- | --- | --- | --- |
| 0:00-0:25 | Freeze contracts | Read baseline | Read baseline | Read baseline | Prepare test harness |
| 0:25-1:25 | Update scope docs | Fixtures and DSP tests | Feature/training benchmark | UI fixture strategy | Test scaffolding |
| 1:25-2:25 | Review interfaces | PSD/spectrum enhancement | Held-out evaluation | Inspector/feed improvements | Run integration checks |
| 2:25-3:25 | Backlog and handoffs | Stabilize backend | Stabilize model output | Add export/filter/error states | Merge backend work |
| 3:25-4:10 | Update docs from changes | Final backend evidence | Final metrics | AppTest/manual checklist | Merge UI and run full suite |
| 4:10-4:30 | Demo script | Sign off DSP | Sign off ML | Sign off UX | Release gate |

## Handoff template

Every account should return this information in its branch description:

```text
Role:
Files changed:
Contract changes: none / list them
Behavior added:
Tests run and exact command:
Observed timing:
Known limitations:
Follow-up needed from another account:
```

## Merge rules

- No generated `__pycache__`, model, notebook output, or temporary files.
- No silent changes to labels, feature order, DataFrame columns, or default units.
- No dependency additions without documenting why they are needed.
- Prefer small commits grouped by one workstream.
- If a branch fails a test, keep it unmerged and report the failure with the smallest reproduction.
- Opus-5 owns the final `git diff --check`, test run, and clean-install verification.
