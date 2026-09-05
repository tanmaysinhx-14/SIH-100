# System Workflow & Development Blueprint

## Part 1: System Runtime Workflow
*This details the step-by-step data pipeline when the application is actively running.*

### 1. Data Ingestion (The Spectrum Generator)
*   **Action:** `simulator.py` continuously produces batches of synthetic RF data (In-Phase and Quadrature samples).
*   **Output:** 10 distinct frequency channels, each populated with either pure background noise, civilian signals (Wi-Fi/FM), or hostile signatures (Radar pulses, Jamming).

### 2. Energy Detection (The Scanner)
*   **Action:** The system calculates the Power Spectral Density (PSD) using a Fast Fourier Transform (FFT) across all 10 channels.
*   **Logic:** Compares the energy of each channel against a pre-defined ambient noise floor threshold.

### 3. Intelligent Prioritization (The Smart Queue)
*   **Action:** Channels that exceed the noise floor are extracted.
*   **Logic:** Instead of scanning sequentially (1 through 10), the system ranks active channels by peak energy and bandwidth, pushing the highest-energy anomalies to the front of the processing queue.

### 4. Machine Learning Classification (The Threat Engine)
*   **Action:** `classifier.py` runs a feature extraction pipeline on the prioritized channels (calculating Peak-to-Average Power Ratio, Spectral Flatness, and Bandwidth).
*   **Logic:** These features are fed into a pre-trained Scikit-Learn Random Forest model.
*   **Output:** The model returns a classification label (`Hostile Radar`, `Hostile Jammer`, `Civilian`, `Noise`) and a confidence percentage (e.g., 94%).

### 5. UI Presentation (The Analyst Dashboard)
*   **Action:** The Streamlit frontend (`app.py`) retrieves the categorized signals.
*   **Logic:** 
    *   Updates a live waterfall/heatmap visualization.
    *   Pushes `Hostile` classifications to the top of a red-highlighted Threat Feed.
    *   Allows the analyst to click a specific alert to render its 2D spectrogram and raw waveform.

---

## Part 2: The 4.5-Hour Development Blueprint
*This is the strict execution plan tailored for AI code generators (Codex, Claude) to build the MVP sequentially.*

### Phase 1: Simulation & DSP Foundation (Minutes 0 - 60)
**Goal:** Generate the fake data and measure its energy.
*   **Prompt AI to:** Create `simulator.py`. Implement functions that return 1D NumPy arrays of complex numbers representing the 4 target classes (Noise, FM, Pulsed Radar, Jammer).
*   **Prompt AI to:** Create `scanner.py`. Write a function that takes the simulator output, runs `scipy.fft.fft`, and returns a dictionary of channels ranked by total energy.

### Phase 2: The Machine Learning Engine (Minutes 60 - 135)
**Goal:** Extract features and train a fast classifier.
*   **Prompt AI to:** Create `classifier.py`. Write a function to extract statistical features (variance, kurtosis, peak power, spectral flatness) from the raw NumPy arrays.
*   **Prompt AI to:** Write a script that generates 1,000 synthetic samples using `simulator.py`, extracts features, and trains a `sklearn.ensemble.RandomForestClassifier`. Save the trained model to memory or a `.joblib` file.

### Phase 3: The Dashboard UI (Minutes 135 - 225)
**Goal:** Build the interactive interface using Streamlit and Plotly.
*   **Prompt AI to:** Create `app.py`. Set up a standard Streamlit layout (`st.set_page_config(layout="wide")`).
*   **Prompt AI to:** Build three UI components: 
    1. A wide Plotly line chart acting as the live spectrum viewer.
    2. A Pandas dataframe rendered via `st.dataframe` for the Threat Alerts.
    3. A sidebar or expander using `scipy.signal.spectrogram` and `matplotlib`/`plotly` to show the spectrogram of a selected signal.

### Phase 4: Integration & State Management (Minutes 225 - 270)
**Goal:** Connect the backend to the frontend and manage the live update loop.
*   **Prompt AI to:** Implement `st.session_state` to hold the threat log history.
*   **Prompt AI to:** Add a `st_autorefresh` component or a "Scan Next Batch" button to trigger the simulator -> scanner -> classifier loop, dynamically updating the charts.