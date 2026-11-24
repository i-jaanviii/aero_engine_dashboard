"""
SIH26054 | AI-Enabled Predictive Maintenance for Aero Engines

DEMO NOTICE:
All engine readings, RUL values, health classifications, and model metrics in this
starter dashboard are simulated for interface demonstration. They are NOT outputs
from a trained model and must not be presented as real model predictions.

Future integration points:
- Replace generate_engine_data() with loading/preprocessing of the actual dataset.
- Replace estimate_rul_demo() with inference from the trained model.
- Replace get_demo_metrics() with metrics calculated on a held-out test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------- Page configuration -----------------------------

st.set_page_config(
    page_title="Aero Engine | Predictive Maintenance",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0B1524"
PANEL = "#111F32"
CYAN = "#38BDF8"
MUTED = "#9AAEC3"
GRID = "#26384D"
TEXT = "#E8F0F7"

# -------------------------------- Demo data -----------------------------------

def generate_engine_data(engine_id: int, current_cycle: int, points: int = 120) -> pd.DataFrame:
    """Generate repeatable simulated sensor history for one demo engine."""
    rng = np.random.default_rng(seed=engine_id * 1009 + current_cycle)
    cycles = np.linspace(max(1, current_cycle - points + 1), current_cycle, points)
    progress = np.linspace(0, 1, points)
    engine_offset = (engine_id % 11) / 100

    temperature = 650 + 24 * progress + engine_offset * 30 + rng.normal(0, 2.2, points)
    pressure = 42 - 3.2 * progress - engine_offset * 2 + rng.normal(0, 0.45, points)
    vibration = 2.0 + 1.45 * progress + engine_offset * 0.8 + rng.normal(0, 0.12, points)
    fuel_flow = 0.82 + 0.12 * progress + engine_offset * 0.05 + rng.normal(0, 0.015, points)

    return pd.DataFrame({
        "Cycle": cycles.astype(int),
        "Exhaust Gas Temperature (°C)": temperature,
        "Pressure Ratio": pressure,
        "Vibration (mm/s)": vibration,
        "Fuel Flow (kg/s)": fuel_flow,
    })
def estimate_rul_demo(engine_id: int, current_cycle: int) -> int:
    """
    Placeholder only. This is a deterministic simulated value, not ML inference.
    Replace this function with the trained model's predict() call.
    """
    simulated_remaining = 115 - (engine_id % 13) * 4 - int(current_cycle * 0.08)
    return int(np.clip(simulated_remaining, 8, 140))


def get_health_status(rul: int) -> tuple[str, str]:
    """Map the demo RUL value to a simple illustrative health category."""
    if rul >= 75:
        return "Healthy", "#22C55E"
    if rul >= 35:
        return "Degraded", "#F59E0B"
    return "Critical", "#EF4444"

def get_demo_metrics() -> dict[str, float]:
    """
    Illustrative placeholder metrics. Not calculated from a trained model/test set.
    Replace with evaluation results from a held-out labelled test dataset.
    """
    return {
        "RMSE": 12.5,
        "MAE": 8.4,
        "Asymmetric score": 150.2,
    }


# ------------------------------ Plot functions --------------------------------

def apply_chart_theme(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=18, r=18, t=28, b=18),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, title_font_color=MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, title_font_color=MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor=NAVY, font_color=TEXT),
    )
    return fig

def sensor_chart(data: pd.DataFrame, sensor: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["Cycle"], y=data[sensor], mode="lines",
        name=sensor, line=dict(color=CYAN, width=2.4),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
    ))
    fig.update_layout(title=sensor, xaxis_title="Operating cycle", yaxis_title=sensor)
    return apply_chart_theme(fig)


def degradation_chart(data: pd.DataFrame) -> go.Figure:
    # Illustrative normalized degradation index, not a measured health score.
    progress = np.linspace(0, 1, len(data))
    degradation = 100 - 58 * (progress ** 1.35)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["Cycle"], y=degradation, mode="lines",
        name="Simulated degradation index",
        line=dict(color="#A78BFA", width=3),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.10)",
    ))
    fig.update_layout(
        title="Historical degradation trend (simulated)",
        xaxis_title="Operating cycle",
        yaxis_title="Illustrative health index",
        yaxis=dict(range=[0, 105]),
    )
    return apply_chart_theme(fig, 330)

# ----------------------------------- UI ----------------------------------------

st.markdown(
    f"""
    <style>
    .stApp {{ background: {NAVY}; color: {TEXT}; }}
    [data-testid="stSidebar"] {{ background: #08111E; border-right: 1px solid {GRID}; }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1500px; }}
    .eyebrow {{ color: {CYAN}; font-size: 0.76rem; letter-spacing: 0.16em; font-weight: 700; }}
    .main-title {{ color: {TEXT}; font-size: 2rem; font-weight: 700; margin: 0.15rem 0 0.2rem 0; }}
    .sub-title {{ color: {MUTED}; font-size: 0.95rem; margin-bottom: 1.25rem; }}
    .demo-banner {{ background: #302515; border: 1px solid #8B6426; color: #F8D99A;
                    padding: 0.85rem 1rem; border-radius: 8px; margin: 0.8rem 0 1.3rem 0; }}
    .metric-card {{ background: {PANEL}; border: 1px solid {GRID}; border-radius: 10px;
                    padding: 1rem 1.1rem; min-height: 112px; }}
    .metric-label {{ color: {MUTED}; font-size: 0.82rem; margin-bottom: 0.45rem; }}
    .metric-value {{ color: {TEXT}; font-size: 1.8rem; font-weight: 700; line-height: 1.15; }}
    .section-heading {{ color: {TEXT}; font-size: 1.08rem; font-weight: 650; margin: 0.35rem 0 0.7rem 0; }}
    div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label {{ color: {MUTED}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">SIH26054  /  ENGINEERING MONITORING SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Aero Engine Predictive Maintenance</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Engine condition monitoring · Remaining useful life overview · Sensor trends</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="demo-banner"><strong>DEMO MODE — SIMULATED DATA</strong><br>'
    'All displayed sensor readings, RUL values, health states and model metrics are illustrative '
    'placeholders. No trained AI model is connected, and these values are not operational predictions.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Monitoring controls")
    engine_id = st.selectbox("Select engine", options=list(range(1, 101)), index=0,
                             format_func=lambda x: f"Engine {x}")
    current_cycle = st.slider("Current operating cycle", min_value=1, max_value=500,
                              value=150, step=1)
    st.divider()
    st.caption("Data source")
    st.markdown("**Simulated demonstration dataset**")
    st.caption("Model integration: not connected")


sensor_data = generate_engine_data(engine_id, current_cycle)
rul = estimate_rul_demo(engine_id, current_cycle)
health, health_color = get_health_status(rul)
metrics = get_demo_metrics()

# Top-level operational indicators
st.markdown('<div class="section-heading">Engine overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Selected engine</div>'
                f'<div class="metric-value">ENG-{engine_id:03d}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Current operating cycle</div>'
                f'<div class="metric-value">{current_cycle:,}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Illustrative RUL</div>'
                f'<div class="metric-value">{rul} <span style="font-size:0.9rem;color:{MUTED}">cycles</span></div></div>',
                unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Demo health status</div>'
                f'<div class="metric-value" style="color:{health_color}">{health}</div></div>',
                unsafe_allow_html=True)

st.caption("RUL and health status above are generated by a placeholder demonstration function, not a trained model.")

st.divider()
left, right = st.columns([1.25, 1])
with left:
    st.markdown('<div class="section-heading">Sensor monitoring</div>', unsafe_allow_html=True)
    selected_sensor = st.selectbox(
        "Sensor channel",
        options=[
            "Exhaust Gas Temperature (°C)",
            "Pressure Ratio",
            "Vibration (mm/s)",
            "Fuel Flow (kg/s)",
        ],
        label_visibility="collapsed",
    )
    st.plotly_chart(sensor_chart(sensor_data, selected_sensor), use_container_width=True)
with right:
    st.markdown('<div class="section-heading">Historical degradation</div>', unsafe_allow_html=True)
    st.plotly_chart(degradation_chart(sensor_data), use_container_width=True)

st.divider()
st.markdown('<div class="section-heading">Model performance metrics</div>', unsafe_allow_html=True)
st.caption("Illustrative placeholder values only — not measured evaluation results.")
m1, m2, m3 = st.columns(3)
for col, label, value, detail in [
    (m1, "RMSE", metrics["RMSE"], "Root Mean Squared Error"),
    (m2, "MAE", metrics["MAE"], "Mean Absolute Error"),
    (m3, "Asymmetric score", metrics["Asymmetric score"], "Illustrative scoring value"),
]:
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value:.2f}</div>'
            f'<div style="color:{MUTED};font-size:0.75rem;margin-top:0.35rem">{detail}</div></div>',
            unsafe_allow_html=True,
        )

with st.expander("View simulated sensor data"):
    st.dataframe(sensor_data.tail(15), use_container_width=True, hide_index=True)

with st.expander("Model and dataset integration notes"):
    st.markdown(
        """
        **Current implementation**
        - Sensor history is generated locally by `generate_engine_data()`.
        - RUL is a placeholder from `estimate_rul_demo()`.
        - Health status is a demonstration rule based on the placeholder RUL.
        - Metrics are fixed illustrative values from `get_demo_metrics()`.

        **When the actual model and dataset are ready**
        1. Load and preprocess the selected engine's time-series records.
        2. Pass the latest sensor window to the trained model.
        3. Replace the placeholder RUL function with model inference.
        4. Calculate RMSE, MAE and the required asymmetric score on a held-out labelled test set.
        5. Display the dataset source, model version and evaluation split in the dashboard.
        """
    )

st.markdown(
    f'<div style="border-top:1px solid {GRID};padding-top:0.8rem;margin-top:1.5rem;'
    f'color:{MUTED};font-size:0.75rem">AERO ENGINE MONITORING SYSTEM · SIH26054 · '
    f'Demonstration interface only</div>',
    unsafe_allow_html=True,
)
