# DSAS Documentation Hub

This folder documents the Defensive Spectrum-Awareness System (DSAS): a synthetic, software-only RF monitoring prototype. It is written for someone who is new to the project and needs to understand what exists, what happens during a scan, and what to build next.

## Read in this order

1. [Current scope](current_scope.md) - what is implemented today, what the data looks like, and what the dashboard actually does.
2. [System architecture](architecture.md) - how the modules connect.
3. [Runtime workflow](workflow.md) - what happens during one sweep.
4. [Execution plan](execution_plan.md) - the order, gates, and time boxes for the next implementation pass.
5. [Feature backlog](feature_backlog.md) - the recommended new work, separated into demo-critical, analyst, and production tracks.
6. [Five-account work split](opus_team_plan.md) - how to divide the work among five Opus accounts without merge conflicts.
7. [Prototype prompts](prototype_prompts.md) - copy-paste briefs for each account.
8. [Verification plan](verification_plan.md) - tests and manual checks that prove the system works.

## Source-of-truth rule

When documents disagree, use this order:

1. The Python source and `requirements.txt`.
2. `current_scope.md`.
3. `architecture.md` and `workflow.md`.
4. The original problem statement and blueprint documents.

The original documents are retained because they explain the motivation and initial idea. They are not a complete record of the current implementation.

## Current implementation at a glance

| Area | Current state |
| --- | --- |
| Input | Synthetic complex IQ only; no SDR hardware |
| Channels | 10 by default, 1,024 samples/channel, 1 MHz sample rate |
| Signal types | Background noise, civilian broadcast, hostile radar, hostile jammer |
| Energy measurement | Welch PSD with a 256-sample segment and mean channel power |
| Prioritization | Power excess above -10 dB by default, multiplied by threat severity |
| Classifier | StandardScaler plus 100-tree Random Forest, max depth 10 |
| Dashboard | Streamlit with spectrum, threat feed, incident log, waveform, and STFT views |
| Storage | In-memory session state; incidents disappear when the app process restarts |

## Run the project

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

For a backend-only smoke check:

```powershell
python simulator.py
python scanner.py
python classifier.py
```

The project is a defensive simulation and visualization prototype. It must not be presented as a validated electronic-warfare detector or as a substitute for certified spectrum-monitoring equipment.
