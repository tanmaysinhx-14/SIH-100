import time
import numpy as np
import pandas as pd
import scipy.signal as signal
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Defensive Spectrum Awareness System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# MODULE INTEGRATION WITH FALLBACK ENGINE
# ==========================================
try:
    from simulator import generate_spectrum_batch
    from scanner import scan_and_prioritize
    from classifier import classify_channel
    HAS_CUSTOM_MODULES = True
except ImportError:
    HAS_CUSTOM_MODULES = False


def fallback_generate_spectrum(num_channels=10, num_samples=1024, sample_rate=1e6):
    """Generates synthetic IQ data across multiple frequency channels."""
    channels = {}
    t = np.arange(num_samples) / sample_rate
    
    for ch in range(num_channels):
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.1
        rand_val = np.random.rand()
        
        if rand_val < 0.45:
            sig_type = "Background Noise"
            iq = noise
        elif rand_val < 0.70:
            sig_type = "Civilian Broadcast"
            iq = np.exp(1j * 2 * np.pi * 50e3 * t) * 0.8 + noise
        elif rand_val < 0.88:
            sig_type = "Hostile Radar"
            pulse_mask = (t * 1e4).astype(int) % 10 < 3
            iq = np.exp(1j * 2 * np.pi * 200e3 * t) * pulse_mask * 3.5 + noise
        else:
            sig_type = "Hostile Jammer"
            iq = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 2.8

        channels[ch] = {
            "iq": iq,
            "type": sig_type,
            "freq_mhz": 100 + ch * 15,
            "sample_rate": sample_rate
        }
    return channels


def fallback_analyze_channels(channels, threshold_db=-10.0):
    """Calculates power spectral density, prioritizes channels, and classifies signals."""
    records = []
    for ch, data in channels.items():
        iq = data["iq"]
        power = np.mean(np.abs(iq) ** 2)
        power_db = 10 * np.log10(power + 1e-12)
        
        sig_type = data["type"]
        if power_db < threshold_db:
            threat_level = "LOW"
            status = "CLEAR"
        elif sig_type == "Hostile Jammer":
            threat_level = "CRITICAL"
            status = "BROADBAND JAMMING"
        elif sig_type == "Hostile Radar":
            threat_level = "HIGH"
            status = "PULSED RADAR LOCK"
        elif sig_type == "Civilian Broadcast":
            threat_level = "LOW"
            status = "STANDARD COMM"
        else:
            threat_level = "MEDIUM"
            status = "ANOMALOUS ACTIVITY"

        priority_score = max(0.0, float(power_db - threshold_db))

        records.append({
            "Channel": ch,
            "Center Freq (MHz)": data["freq_mhz"],
            "Power (dB)": round(power_db, 2),
            "Classification": sig_type,
            "Threat Level": threat_level,
            "Status": status,
            "Priority Score": round(priority_score, 2),
            "Timestamp": time.strftime("%H:%M:%S")
        })
        
    df = pd.DataFrame(records)
    df = df.sort_values(by="Priority Score", ascending=False).reset_index(drop=True)
    return df


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "threat_log" not in st.session_state:
    st.session_state.threat_log = pd.DataFrame()
if "current_channels" not in st.session_state:
    st.session_state.current_channels = fallback_generate_spectrum()
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = fallback_analyze_channels(st.session_state.current_channels)


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🛡️ Control Panel")
st.sidebar.markdown("---")

st.sidebar.subheader("Scan Settings")
threshold_db = st.sidebar.slider("Noise Floor Threshold (dB)", -25.0, 10.0, -10.0, 1.0)
auto_refresh = st.sidebar.checkbox("Enable Auto Scanning", value=False)
scan_interval = st.sidebar.slider("Scan Speed (seconds)", 1, 5, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("Actions")

if st.sidebar.button("⚡ Execute Immediate Sweep", use_container_width=True):
    if HAS_CUSTOM_MODULES:
        st.session_state.current_channels = generate_spectrum_batch()
        st.session_state.current_analysis = scan_and_prioritize(st.session_state.current_channels, threshold_db)
    else:
        st.session_state.current_channels = fallback_generate_spectrum()
        st.session_state.current_analysis = fallback_analyze_channels(
            st.session_state.current_channels, threshold_db
        )

if st.sidebar.button("🗑️ Clear Threat History", use_container_width=True):
    st.session_state.threat_log = pd.DataFrame()
    st.rerun()

st.sidebar.markdown("---")
if HAS_CUSTOM_MODULES:
    st.sidebar.success("Backend Engine: Custom Modules Connected")
else:
    st.sidebar.info("Backend Engine: Standalone Generator Active")


# ==========================================
# MAIN DASHBOARD HEADER
# ==========================================
st.title("📡 Defensive Spectrum-Awareness System")
st.caption("Real-Time RF Monitoring, Energy Prioritization & Threat Identification Engine")

analysis_df = st.session_state.current_analysis

# Update persistent threat history log
threats_detected = analysis_df[analysis_df["Threat Level"].isin(["HIGH", "CRITICAL"])]
if not threats_detected.empty:
    st.session_state.threat_log = pd.concat(
        [threats_detected, st.session_state.threat_log]
    ).drop_duplicates().head(50)

# Key Performance Indicators (KPIs)
col1, col2, col3, col4 = st.columns(4)
total_channels = len(analysis_df)
active_channels = len(analysis_df[analysis_df["Priority Score"] > 0])
critical_threats = len(analysis_df[analysis_df["Threat Level"] == "CRITICAL"])
high_threats = len(analysis_df[analysis_df["Threat Level"] == "HIGH"])

col1.metric("Monitored Channels", total_channels)
col2.metric("Signals Above Threshold", active_channels)
col3.metric("Critical Threat (Jammers)", critical_threats, delta_color="inverse")
col4.metric("High Threat (Radars)", high_threats, delta_color="inverse")

st.markdown("---")

# ==========================================
# TABS INTERFACE
# ==========================================
tab_spectrum, tab_threats, tab_inspector = st.tabs([
    "📊 Live Spectrum Monitor", 
    "🚨 Prioritized Threat Feed", 
    "🔍 Signal Inspector & STFT"
])


# ------------------------------------------
# TAB 1: LIVE SPECTRUM MONITOR
# ------------------------------------------
with tab_spectrum:
    st.subheader("Multi-Channel Power Spectral Density")
    
    # Spectrum Power Chart
    fig_spectrum = px.bar(
        analysis_df.sort_values(by="Channel"),
        x="Center Freq (MHz)",
        y="Power (dB)",
        color="Threat Level",
        color_discrete_map={
            "CRITICAL": "#FF2B2B",
            "HIGH": "#FF8C00",
            "MEDIUM": "#FFD700",
            "LOW": "#00CC96"
        },
        hover_data=["Channel", "Classification", "Status", "Priority Score"],
        title="Current Spectrum Scan across Bands"
    )
    
    fig_spectrum.add_hline(
        y=threshold_db, 
        line_dash="dash", 
        line_color="red", 
        annotation_text=f"Threshold ({threshold_db} dB)"
    )
    fig_spectrum.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_spectrum, use_container_width=True)

    # Historical Waterfall Simulation
    st.subheader("Spectrum Heatmap (Channel Power over Channels)")
    power_matrix = np.tile(analysis_df.sort_values(by="Channel")["Power (dB)"].values, (10, 1))
    
    fig_waterfall = px.imshow(
        power_matrix,
        labels=dict(x="Channel Index", y="Sweep Time Window", color="Power (dB)"),
        x=analysis_df.sort_values(by="Channel")["Channel"].values,
        color_continuous_scale="Viridis"
    )
    fig_waterfall.update_layout(template="plotly_dark", height=250)
    st.plotly_chart(fig_waterfall, use_container_width=True)


# ------------------------------------------
# TAB 2: PRIORITIZED THREAT FEED
# ------------------------------------------
with tab_threats:
    st.subheader("Prioritized Real-Time Scan Observations")
    st.caption("Channels are ranked dynamically by priority score (energy above noise baseline).")
    
    # Custom DataFrame Styling
    def highlight_threats(val):
        if val == "CRITICAL":
            return "background-color: #721c24; color: white; font-weight: bold;"
        elif val == "HIGH":
            return "background-color: #856404; color: white; font-weight: bold;"
        elif val == "MEDIUM":
            return "background-color: #383d41; color: white;"
        return "background-color: #155724; color: white;"

    styled_df = analysis_df.style.map(highlight_threats, subset=["Threat Level"])
    st.dataframe(styled_df, use_container_width=True, height=300)

    st.subheader("Threat Incident Log (History)")
    if not st.session_state.threat_log.empty:
        st.dataframe(st.session_state.threat_log, use_container_width=True)
    else:
        st.info("No critical or high-priority threats logged in history.")


# ------------------------------------------
# TAB 3: SIGNAL INSPECTOR & STFT SPECTROGRAM
# ------------------------------------------
with tab_inspector:
    st.subheader("Deep Signal Inspection")
    selected_ch = st.selectbox("Select Channel to Inspect:", list(st.session_state.current_channels.keys()))
    
    ch_data = st.session_state.current_channels[selected_ch]
    iq_samples = ch_data["iq"]
    fs = ch_data["sample_rate"]
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"**Channel {selected_ch} IQ Waveform (In-Phase / Quadrature)**")
        time_axis = np.arange(len(iq_samples)) / fs * 1e6  # Microseconds
        
        fig_iq = go.Figure()
        fig_iq.add_trace(go.Scatter(x=time_axis, y=np.real(iq_samples), name="In-Phase (I)", line=dict(color="#00BFFF")))
        fig_iq.add_trace(go.Scatter(x=time_axis, y=np.imag(iq_samples), name="Quadrature (Q)", line=dict(color="#FF6347")))
        fig_iq.update_layout(
            template="plotly_dark", 
            xaxis_title="Time (µs)", 
            yaxis_title="Amplitude", 
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_iq, use_container_width=True)

    with col_b:
        st.markdown(f"**Channel {selected_ch} Spectrogram (STFT)**")
        f, t_spec, Sxx = signal.spectrogram(iq_samples, fs=fs, nperseg=64)
        Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)
        
        fig_spec = px.imshow(
            Sxx_db,
            x=t_spec * 1e6,
            y=f / 1e3,
            labels=dict(x="Time (µs)", y="Frequency (kHz)", color="Power (dB)"),
            color_continuous_scale="Jet",
            origin="lower"
        )
        fig_spec.update_layout(
            template="plotly_dark", 
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_spec, use_container_width=True)

# ==========================================
# AUTO REFRESH LOOP
# ==========================================
if auto_refresh:
    time.sleep(scan_interval)
    if HAS_CUSTOM_MODULES:
        st.session_state.current_channels = generate_spectrum_batch()
        st.session_state.current_analysis = scan_and_prioritize(st.session_state.current_channels, threshold_db)
    else:
        st.session_state.current_channels = fallback_generate_spectrum()
        st.session_state.current_analysis = fallback_analyze_channels(
            st.session_state.current_channels, threshold_db
        )
    st.rerun()