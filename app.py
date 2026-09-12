"""Streamlit operations dashboard for the synthetic DSAS pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.signal as signal
import streamlit as st

from classifier import _CLASSIFIER_INSTANCE
from scanner import (
    HAS_CLASSIFIER,
    calculate_psd_metrics,
    estimate_noise_floor,
    get_priority_breakdown,
    scan_and_prioritize,
)
from simulator import SyntheticChannelSource, generate_spectrum_batch


st.set_page_config(
    page_title="Defensive Spectrum-Awareness System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _new_scan(threshold_db: float) -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    """Generate and analyze one complete synthetic spectrum sweep."""
    source = SyntheticChannelSource(num_channels=10)
    channels = source.read_batch()
    analysis = scan_and_prioritize(channels, threshold_db=threshold_db)
    return channels, analysis


def _ensure_state(threshold_db: float) -> None:
    if "threat_log" not in st.session_state:
        st.session_state.threat_log = pd.DataFrame()
    if "threshold_slider" not in st.session_state:
        st.session_state.threshold_slider = threshold_db
    if "current_channels" not in st.session_state:
        channels, analysis = _new_scan(threshold_db)
        st.session_state.current_channels = channels
        st.session_state.current_analysis = analysis
        st.session_state.analysis_threshold = threshold_db
        st.session_state.scan_number = 1
    elif st.session_state.get("analysis_threshold") != threshold_db:
        st.session_state.current_analysis = scan_and_prioritize(
            st.session_state.current_channels,
            threshold_db=threshold_db,
        )
        st.session_state.analysis_threshold = threshold_db


def _execute_sweep(threshold_db: float) -> None:
    channels, analysis = _new_scan(threshold_db)
    st.session_state.current_channels = channels
    st.session_state.current_analysis = analysis
    st.session_state.analysis_threshold = threshold_db
    st.session_state.scan_number = int(st.session_state.get("scan_number", 0)) + 1


def _append_threats(analysis: pd.DataFrame) -> None:
    if st.session_state.get("log_clear_scan") == st.session_state.get("scan_number"):
        return
    threats = analysis[analysis["Threat Level"].isin(["HIGH", "CRITICAL"])].copy()
    if threats.empty:
        return
    scan_number = st.session_state.get("scan_number", 1)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    threats.insert(
        0,
        "Incident ID",
        [f"S{scan_number:04d}-CH{channel}" for channel in threats["Channel"]],
    )
    threats.insert(1, "Detected At", now)
    existing = st.session_state.threat_log
    st.session_state.threat_log = (
        pd.concat([threats, existing], ignore_index=True)
        .drop_duplicates(subset=["Incident ID"], keep="first")
        .head(50)
    )


def _render_spectrum(
    channels: dict[int, dict[str, Any]], analysis: pd.DataFrame, threshold_db: float
) -> None:
    st.subheader("Multi-Channel Power Spectrum")
    ordered = analysis.sort_values("Channel")
    spectrum_figure = px.bar(
        ordered,
        x="Center Freq (MHz)",
        y="Power (dB)",
        color="Threat Level",
        color_discrete_map={
            "CRITICAL": "#ff3b30",
            "HIGH": "#ff9500",
            "MEDIUM": "#ffd60a",
            "LOW": "#30d158",
        },
        hover_data=["Channel", "Classification", "Confidence (%)", "Priority Score"],
        title="Current power by monitored frequency channel",
    )
    spectrum_figure.add_hline(
        y=threshold_db,
        line_dash="dash",
        line_color="#ff453a",
        annotation_text=f"Noise floor ({threshold_db:.0f} dB)",
    )
    spectrum_figure.update_layout(template="plotly_dark", height=380, margin=dict(t=55))
    st.plotly_chart(spectrum_figure, use_container_width=True)

    with st.expander("🌐 True Frequency-Bin PSD Curves Across Channels", expanded=False):
        psd_fig = go.Figure()
        for channel_id, ch in sorted(channels.items()):
            iq = np.asarray(ch["iq"], dtype=np.complex128)
            sample_rate = float(ch.get("sample_rate", 1.0e6))
            center_mhz = float(ch.get("center_freq_mhz", 100.0 + channel_id * 15.0))
            metrics = calculate_psd_metrics(iq, sample_rate)
            freqs_mhz = center_mhz + (metrics["freqs"] / 1.0e6)
            psd_fig.add_trace(
                go.Scatter(
                    x=freqs_mhz,
                    y=metrics["psd_db"],
                    mode="lines",
                    name=f"Ch {channel_id} ({ch.get('type', 'Signal')})",
                )
            )
        psd_fig.add_hline(
            y=threshold_db,
            line_dash="dash",
            line_color="#ff453a",
            annotation_text=f"Threshold ({threshold_db:.1f} dB)",
        )
        psd_fig.update_layout(
            template="plotly_dark",
            title="Frequency-Bin Power Spectral Density (Welch PSD)",
            xaxis_title="Frequency (MHz)",
            yaxis_title="PSD (dB/Hz)",
            height=380,
        )
        st.plotly_chart(psd_fig, use_container_width=True)

    st.subheader("Spectrum Waterfall History")
    current_power = ordered["Power (dB)"].to_numpy(dtype=float)
    waterfall_history = st.session_state.get("waterfall_history", [])
    waterfall_history = (waterfall_history + [current_power])[-12:]
    st.session_state.waterfall_history = waterfall_history
    waterfall = np.vstack(waterfall_history)
    waterfall_figure = px.imshow(
        waterfall,
        x=ordered["Channel"].tolist(),
        labels={"x": "Channel", "y": "Sweep History", "color": "Power (dB)"},
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    waterfall_figure.update_layout(template="plotly_dark", height=280)
    st.plotly_chart(waterfall_figure, use_container_width=True)


def _render_threat_feed(analysis: pd.DataFrame) -> None:
    st.subheader("Prioritized Real-Time Scan Observations")
    st.caption(
        "Priority combines energy above the noise floor with model-assigned threat severity."
    )

    # Filtering Controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        threat_filter = st.selectbox(
            "Filter by Threat Level",
            options=["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
        )
    with f_col2:
        class_filter = st.selectbox(
            "Filter by Classification",
            options=[
                "All",
                "Hostile Jammer",
                "Hostile Radar",
                "Civilian Broadcast",
                "Background Noise",
            ],
        )
    with f_col3:
        min_priority = st.number_input(
            "Min Priority Score", min_value=0.0, max_value=100.0, value=0.0, step=1.0
        )

    filtered_df = analysis.copy()
    if threat_filter != "All":
        filtered_df = filtered_df[filtered_df["Threat Level"] == threat_filter]
    if class_filter != "All":
        filtered_df = filtered_df[filtered_df["Classification"] == class_filter]
    filtered_df = filtered_df[filtered_df["Priority Score"] >= min_priority]

    def row_style(row: pd.Series) -> list[str]:
        colors = {
            "CRITICAL": "background-color: #5c1010; color: white; font-weight: bold",
            "HIGH": "background-color: #6b3d00; color: white; font-weight: bold",
            "MEDIUM": "background-color: #544b00; color: white",
            "LOW": "",
        }
        return [colors.get(row["Threat Level"], "")] * len(row)

    st.dataframe(
        filtered_df.style.apply(row_style, axis=1),
        use_container_width=True,
        height=320,
    )

    # Export Buttons
    e_col1, e_col2, e_col3, e_col4 = st.columns(4)
    with e_col1:
        csv_obs = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Observations (CSV)",
            data=csv_obs,
            file_name=f"dsas_observations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e_col2:
        json_obs = filtered_df.to_json(orient="records", indent=2).encode("utf-8")
        st.download_button(
            "📥 Export Observations (JSON)",
            data=json_obs,
            file_name=f"dsas_observations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Threat Incident Log")
    if st.session_state.threat_log.empty:
        st.info("No HIGH or CRITICAL incidents have been logged yet.")
    else:
        st.dataframe(st.session_state.threat_log, use_container_width=True, height=260)
        with e_col3:
            csv_inc = st.session_state.threat_log.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Incidents (CSV)",
                data=csv_inc,
                file_name=f"dsas_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with e_col4:
            json_inc = st.session_state.threat_log.to_json(
                orient="records", indent=2
            ).encode("utf-8")
            st.download_button(
                "📥 Export Incidents (JSON)",
                data=json_inc,
                file_name=f"dsas_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )


def _render_inspector(
    channels: dict[int, dict[str, Any]],
    analysis: pd.DataFrame,
    threshold_db: float,
) -> None:
    st.subheader("Signal Inspector & STFT Spectrogram")
    selected_channel = st.selectbox(
        "Select channel",
        options=list(channels),
        format_func=lambda channel: (
            f"Channel {channel} — {channels[channel].get('freq_mhz', 0):.1f} MHz"
        ),
    )
    channel = channels[selected_channel]
    ch_analysis = analysis[analysis["Channel"] == selected_channel].iloc[0]

    # Explainable Priority Score Section
    breakdown = get_priority_breakdown(
        power_db=float(ch_analysis["Power (dB)"]),
        threshold_db=threshold_db,
        threat_level=str(ch_analysis["Threat Level"]),
        classification=str(ch_analysis["Classification"]),
    )

    st.info(f"💡 **Explainable Priority Score Breakdown**: {breakdown['explanation']}")

    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
    exp_col1.metric("Measured Power", f"{breakdown['power_db']:.1f} dB")
    exp_col2.metric("Power Excess", f"{breakdown['power_excess_db']:.1f} dB")
    exp_col3.metric("Threat Multiplier", f"{breakdown['threat_multiplier']}x")
    exp_col4.metric("Final Priority Score", f"{breakdown['priority_score']:.1f}")

    iq = np.asarray(channel["iq"], dtype=np.complex128)
    sample_rate = float(channel.get("sample_rate", 1.0e6))
    time_axis_us = np.arange(iq.size) / sample_rate * 1.0e6

    left, right = st.columns(2)
    with left:
        st.markdown("**In-Phase / Quadrature waveform**")
        waveform = go.Figure()
        waveform.add_trace(
            go.Scatter(x=time_axis_us, y=iq.real, name="I", line=dict(color="#00bfff"))
        )
        waveform.add_trace(
            go.Scatter(x=time_axis_us, y=iq.imag, name="Q", line=dict(color="#ff6347"))
        )
        waveform.update_layout(
            template="plotly_dark",
            xaxis_title="Time (µs)",
            yaxis_title="Amplitude",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(waveform, use_container_width=True)

    with right:
        st.markdown("**Short-Time Fourier Transform spectrogram**")
        frequencies, times, spectrogram = signal.spectrogram(
            iq,
            fs=sample_rate,
            nperseg=min(64, iq.size),
            noverlap=min(32, max(0, iq.size // 4)),
            return_onesided=False,
            scaling="density",
        )
        frequencies = np.fft.fftshift(frequencies)
        spectrogram = np.fft.fftshift(np.real(spectrogram), axes=0)
        spectrogram_db = 10.0 * np.log10(np.maximum(spectrogram, np.finfo(float).tiny))
        spectrogram_figure = px.imshow(
            spectrogram_db,
            x=times * 1.0e6,
            y=frequencies / 1.0e3,
            labels={"x": "Time (µs)", "y": "Frequency (kHz)", "color": "Power (dB)"},
            color_continuous_scale="Turbo",
            origin="lower",
            aspect="auto",
        )
        spectrogram_figure.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(spectrogram_figure, use_container_width=True)

    # Model Evaluation Diagnostics Expander
    with st.expander("🤖 Classifier Performance & Held-Out Diagnostics", expanded=False):
        if HAS_CLASSIFIER:
            eval_metrics = _CLASSIFIER_INSTANCE.evaluate_model(samples_per_class=50)
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.write(f"**Held-Out Model Accuracy**: `{eval_metrics['accuracy'] * 100:.1f}%`")
                st.write("**Top Feature Importance Weights**:")
                imp_df = pd.DataFrame(
                    list(eval_metrics["feature_importances"].items()),
                    columns=["Feature", "Importance"],
                ).sort_values("Importance", ascending=False)
                st.dataframe(imp_df, use_container_width=True, height=180)

            with d_col2:
                st.write("**Confusion Matrix**:")
                cm_df = pd.DataFrame(
                    eval_metrics["confusion_matrix"],
                    index=eval_metrics["class_labels"],
                    columns=eval_metrics["class_labels"],
                )
                st.dataframe(cm_df, use_container_width=True, height=180)
        else:
            st.warning("ML Classifier is offline. Fallback heuristic active.")


def _render_main_dashboard(threshold_db: float) -> None:
    analysis = st.session_state.current_analysis
    channels = st.session_state.current_channels
    _append_threats(analysis)

    st.title("🛡️ Defensive Spectrum-Awareness System")
    st.caption("Synthetic RF monitoring, energy prioritization, and threat identification")
    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    kpi_1.metric("Monitored Channels", len(analysis))
    kpi_2.metric("Signals Above Threshold", int((analysis["Priority Score"] > 0).sum()))
    kpi_3.metric(
        "Critical Threats",
        int((analysis["Threat Level"] == "CRITICAL").sum()),
        delta_color="inverse",
    )
    kpi_4.metric(
        "High Threats",
        int((analysis["Threat Level"] == "HIGH").sum()),
        delta_color="inverse",
    )
    st.divider()

    spectrum_tab, threat_tab, inspector_tab = st.tabs(
        ["📊 Live Spectrum Monitor", "🚨 Prioritized Threat Feed", "🔍 Signal Inspector"]
    )
    with spectrum_tab:
        _render_spectrum(channels, analysis, threshold_db)
    with threat_tab:
        _render_threat_feed(analysis)
    with inspector_tab:
        _render_inspector(channels, analysis, threshold_db)


def main() -> None:
    # Handle stateful threshold slider value
    if "threshold_slider" not in st.session_state:
        st.session_state.threshold_slider = -10.0

    threshold_db = st.sidebar.slider(
        "Noise Floor Threshold (dB)",
        min_value=-25.0,
        max_value=10.0,
        value=float(st.session_state.threshold_slider),
        step=1.0,
        key="threshold_slider_widget",
    )
    st.session_state.threshold_slider = threshold_db

    _ensure_state(threshold_db)

    # Auto-Calibration Button
    if st.sidebar.button("⚡ Auto-Calibrate Noise Floor", use_container_width=True):
        calibrated = estimate_noise_floor(st.session_state.current_channels)
        st.session_state.threshold_slider = calibrated
        st.sidebar.success(f"Calibrated threshold to {calibrated:.1f} dB")
        st.rerun()

    auto_refresh = st.sidebar.toggle("Enable Auto-Refresh", value=False)
    scan_interval = st.sidebar.slider("Auto-refresh interval (seconds)", 1, 10, 3)
    sweep_requested = st.sidebar.button("⚡ Manual Sweep", use_container_width=True)
    clear_requested = st.sidebar.button("🗑️ Clear Threat Log", use_container_width=True)

    if sweep_requested:
        _execute_sweep(threshold_db)
    if clear_requested:
        st.session_state.threat_log = pd.DataFrame()
        st.session_state.log_clear_scan = st.session_state.get("scan_number", 1)

    st.sidebar.divider()
    if HAS_CLASSIFIER:
        st.sidebar.success("Classifier: Random Forest online")
    else:
        st.sidebar.warning("Classifier unavailable; heuristic fallback active")
    st.sidebar.caption(f"Sweep #{st.session_state.get('scan_number', 1)}")

    if auto_refresh and hasattr(st, "fragment"):
        @st.fragment(run_every=scan_interval)
        def refreshed_dashboard() -> None:
            _execute_sweep(threshold_db)
            _render_main_dashboard(threshold_db)

        refreshed_dashboard()
    else:
        _render_main_dashboard(threshold_db)
        if auto_refresh:
            st.info("Auto-refresh requires a recent Streamlit version with fragment support.")


if __name__ == "__main__":
    main()
