# DSAS Technical Stack

## Runtime

- Python 3.10 or newer.
- NumPy for complex arrays, random signal generation, and numerical operations.
- SciPy for Welch PSD, periodogram, and STFT/spectrogram operations.
- pandas for scan results and in-memory incident tables.
- scikit-learn for StandardScaler and RandomForestClassifier.
- Plotly for interactive charts.
- Streamlit for the dashboard and session state.

## Dependency policy

The direct dependencies are listed in the repository-root `requirements.txt`. The project must not add SDR drivers, C bindings, GPU frameworks, or third-party refresh components for the MVP.

When adding a dependency:

1. Explain the need in the relevant documentation.
2. Confirm that the capability cannot be implemented with the existing stack.
3. Add a version floor compatible with Python 3.10+.
4. Run a clean-install smoke check.

## Module-to-library map

| Module | Main libraries | Purpose |
| --- | --- | --- |
| `simulator.py` | NumPy | Complex AWGN, phase modulation, radar pulse train, jammer noise |
| `scanner.py` | NumPy, SciPy, pandas | Welch PSD, power measurement, threat-weighted ranking |
| `classifier.py` | NumPy, SciPy, scikit-learn, pickle | Nine features, scaling, Random Forest, explicit save/load |
| `app.py` | NumPy, SciPy, pandas, Plotly, Streamlit | Controls, charts, session state, incident feed |

## Performance targets

- `RFSignalClassifier.train(samples_per_class=300)`: target under 2 seconds and hard limit under 5 seconds on the development machine.
- Default ten-channel scan: interactive manual-sweep latency.
- Waterfall and incident history: bounded in-memory buffers.
- No model file required for normal first-use operation.

## Data and persistence

The default system is in-memory. The trained model can be explicitly saved or loaded as a local pickle file, but generated models are not part of the source repository. The Streamlit incident log lasts only for the current session. A future persistence layer must define retention, export, and privacy behavior before implementation.

## Deployment shape

The current deployment is a single local Streamlit process:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

For CI or automated verification, run Python module smoke checks and Streamlit AppTest. A future hosted deployment needs configuration management, logging, health checks, access control, and resource limits.
