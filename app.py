"""
SIH26054 | AI-Enabled Predictive Maintenance for Aero Engines
NASA C-MAPSS FD001 | Linear Regression pipeline
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Aero Engine Predictive Maintenance",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0B1524"
PANEL = "#111F32"
CYAN = "#38BDF8"
MUTED = "#9AAEC3"
GRID = "#26384D"
TEXT = "#E8F0F7"

MODEL_PATH = "rul_model.joblib"
DATA_PATH = "dashboard_sensor_data.csv"

@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_sensor_data():
    data = pd.read_csv(DATA_PATH)
    data["engine_id"] = data["engine_id"].astype(int)
    data["cycle"] = data["cycle"].astype(int)
    return data
def predict_rul(model_pipeline, sensor_window: pd.DataFrame, selected_sensors: list[str], rul_cap: int) -> float:
    # Preserve the exact sensor order stored with the trained model.
    # Pass raw values; StandardScaler is already inside the saved pipeline.
    raw_features = sensor_window[selected_sensors].to_numpy(dtype=float).reshape(1, -1)
    if raw_features.shape[1] != 450:
        raise ValueError(f"Expected 450 input features, received {raw_features.shape[1]}.")
    prediction = float(model_pipeline.predict(raw_features)[0])
    return float(np.clip(prediction, 0, rul_cap))

def calculate_nasa_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    errors = predicted - actual
    penalties = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0,
    )
    return float(np.sum(penalties))

def evaluate_model(data: pd.DataFrame, bundle: dict) -> tuple[float, float, float]:
    pipeline = bundle["model"]
    sensors = bundle["selected_sensors"]
    window_size = int(bundle["window_size"])
    rul_cap = int(bundle["rul_cap"])

    actual_values = []
    predicted_values = []
    for engine_id in sorted(data["engine_id"].unique()):
        engine = data[data["engine_id"] == engine_id].sort_values("cycle")
        window = engine.tail(window_size)
        if len(window) != window_size:
            continue
        predicted_values.append(predict_rul(pipeline, window, sensors, rul_cap))
        actual_values.append(float(engine["actual_RUL"].iloc[-1]))

    actual = np.asarray(actual_values, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    mae = float(np.mean(np.abs(predicted - actual)))
    score = calculate_nasa_score(actual, predicted)
    return rmse, mae, score

def make_chart_theme(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=24, r=20, t=48, b=24),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, title_font_color=MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, title_font_color=MUTED),
        hoverlabel=dict(bgcolor=NAVY, font_color=TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
try:
    bundle = load_model_bundle()
    sensor_data = load_sensor_data()
except Exception as exc:
    st.error("Could not load the model or sensor dataset.")
    st.exception(exc)
    st.stop()

required_columns = {"engine_id", "cycle", "actual_RUL", *bundle["selected_sensors"]}
missing_columns = required_columns.difference(sensor_data.columns)
if missing_columns:
    st.error(f"Missing required dataset columns: {', '.join(sorted(missing_columns))}")
    st.stop()

selected_sensors = bundle["selected_sensors"]
window_size = int(bundle["window_size"])
rul_cap = int(bundle["rul_cap"])
pipeline = bundle["model"]
# Styling
st.markdown(
    f"""
    <style>
    .stApp {{ background: {NAVY}; color: {TEXT}; }}
    [data-testid="stSidebar"] {{ background: #08111E; border-right: 1px solid {GRID}; }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1550px; }}
    .eyebrow {{ color: {CYAN}; font-size: .76rem; letter-spacing: .15em; font-weight: 700; }}
    .main-title {{ color: {TEXT}; font-size: 2rem; font-weight: 750; margin: .15rem 0 .2rem; }}
    .sub-title {{ color: {MUTED}; margin-bottom: 1.1rem; }}
    .metric-card {{ background: {PANEL}; border: 1px solid {GRID}; border-radius: 10px; padding: 1rem 1.1rem; min-height: 108px; }}
    .metric-label {{ color: {MUTED}; font-size: .82rem; margin-bottom: .45rem; }}
    .metric-value {{ color: {TEXT}; font-size: 1.8rem; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">SIH26054 / ENGINEERING MONITORING SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Aero Engine Predictive Maintenance</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">NASA C-MAPSS FD001 · Engine condition monitoring · Remaining useful life prediction</div>',
    unsafe_allow_html=True,
)
# Sidebar controls
engine_ids = sorted(sensor_data["engine_id"].unique().tolist())
with st.sidebar:
    st.markdown("### Monitoring controls")
    engine_id = st.selectbox(
        "Select engine",
        options=engine_ids,
        format_func=lambda value: f"Engine {value}",
    )

engine_history = sensor_data[sensor_data["engine_id"] == engine_id].sort_values("cycle").copy()
max_cycle = int(engine_history["cycle"].max())

with st.sidebar:
    current_cycle = st.slider(
        "Current operating cycle",
        min_value=window_size,
        max_value=max_cycle,
        value=max_cycle,
        step=1,
    )
    st.divider()
    st.caption("Dataset: NASA C-MAPSS FD001")
    st.caption(f"Model: {bundle.get('model_name', 'Saved regression pipeline')}")
    st.caption(f"Input: {window_size} cycles × {len(selected_sensors)} sensors")

visible_history = engine_history[engine_history["cycle"] <= current_cycle].sort_values("cycle")
latest_window = visible_history.tail(window_size)

if len(latest_window) < window_size:
    st.warning(f"At least {window_size} cycles are required. Only {len(latest_window)} are available.")
    st.stop()

try:
    predicted_rul = predict_rul(pipeline, latest_window, selected_sensors, rul_cap)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

final_actual_rul = float(engine_history["actual_RUL"].iloc[-1])

actual_rul = min(
    rul_cap,
    final_actual_rul + (max_cycle - current_cycle)
)
prediction_error = predicted_rul - actual_rul

if predicted_rul >= 75:
    health_status, health_color = "Healthy", "#22C55E"
elif predicted_rul >= 35:
    health_status, health_color = "Degraded", "#F59E0B"
else:
    health_status, health_color = "Critical", "#EF4444"
# Main engine indicators
st.markdown("### Engine overview")
c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "Selected engine", f"ENG-{engine_id:03d}"),
    (c2, "Current operating cycle", f"{current_cycle:,}"),
    (c3, "Predicted remaining useful life", f"{predicted_rul:.2f} cycles"),
    (c4, "Health status (RUL threshold rule)", health_status),
]
for col, label, value in cards:
    with col:
        color_style = f"color:{health_color};" if label.startswith("Health") else ""
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="{color_style}">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.caption(
    f"Benchmark reference for this FD001 test engine: actual RUL = {actual_rul:.0f} cycles · "
    f"prediction error = {prediction_error:+.2f} cycles. "
    "Actual RUL is shown because this is a labelled benchmark test set."
)

st.divider()
left, right = st.columns([1.1, 1])

with left:
    st.markdown("### Sensor monitoring")
    default_sensor = "sensor_2" if "sensor_2" in selected_sensors else selected_sensors[0]
    selected_sensor = st.selectbox(
        "Sensor channel",
        options=selected_sensors,
        index=selected_sensors.index(default_sensor),
    )
    fig_sensor = go.Figure()
    fig_sensor.add_trace(
        go.Scatter(
            x=visible_history["cycle"],
            y=visible_history[selected_sensor],
            mode="lines",
            name=selected_sensor,
            line=dict(color=CYAN, width=2.3),
        )
    )
    fig_sensor.update_layout(
        title=f"{selected_sensor} — observed sensor trend",
        xaxis_title="Operating cycle",
        yaxis_title="Raw sensor value",
    )
    st.plotly_chart(make_chart_theme(fig_sensor), use_container_width=True)
with right:
    st.markdown("### Historical RUL prediction curve")
    # Re-run inference at historical cycle endpoints using each endpoint's latest 30-cycle window.
    history_predictions = []
    candidate_cycles = list(range(window_size, current_cycle + 1))
    stride = max(1, len(candidate_cycles) // 100)
    sampled_cycles = candidate_cycles[::stride]
    if sampled_cycles[-1] != current_cycle:
        sampled_cycles.append(current_cycle)

    for endpoint in sampled_cycles:
        past_window = engine_history[engine_history["cycle"] <= endpoint].sort_values("cycle").tail(window_size)
        if len(past_window) == window_size:
            history_predictions.append(
                (endpoint, predict_rul(pipeline, past_window, selected_sensors, rul_cap))
            )

    curve_df = pd.DataFrame(history_predictions, columns=["cycle", "predicted_RUL"])
    fig_rul = go.Figure()
    fig_rul.add_trace(
        go.Scatter(
            x=curve_df["cycle"],
            y=curve_df["predicted_RUL"],
            mode="lines+markers",
            name="Model-predicted RUL",
            line=dict(color="#A78BFA", width=2.6),
            marker=dict(size=5),
        )
    )
    fig_rul.update_layout(
        title="Predicted RUL over historical windows",
        xaxis_title="Operating cycle",
        yaxis_title="Predicted remaining cycles",
        yaxis=dict(range=[0, rul_cap + 5]),
    )
    st.plotly_chart(make_chart_theme(fig_rul), use_container_width=True)

st.caption(
    "The historical curve is generated by rerunning the trained model on successive 30-cycle windows. "
    "It is a model-prediction trend, not a separately measured physical degradation index."
)

st.divider()
st.markdown("### Model performance — FD001 test set")
try:
    rmse, mae, asymmetric_score = evaluate_model(sensor_data, bundle)
    m1, m2, m3 = st.columns(3)
    for col, label, value, detail in [
        (m1, "RMSE", rmse, "Root Mean Squared Error · cycles"),
        (m2, "MAE", mae, "Mean Absolute Error · cycles"),
        (m3, "NASA asymmetric score", asymmetric_score, "Lower is better"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value:.3f}</div>'
                f'<div style="color:{MUTED};font-size:.75rem;margin-top:.35rem">{detail}</div></div>',
                unsafe_allow_html=True,
            )
    st.caption("Metrics are recalculated across all 100 labelled FD001 test engines using the saved model.")
except Exception as exc:
    st.error(f"Could not calculate evaluation metrics: {exc}")

with st.expander("View latest 30-cycle model input"):
    st.write("The model receives the raw values below in the exact sensor order stored in the model bundle.")
    st.code(" → ".join(selected_sensors))
    st.dataframe(latest_window[["cycle"] + selected_sensors], use_container_width=True, hide_index=True)

with st.expander("View selected engine sensor records"):
    st.dataframe(visible_history, use_container_width=True, hide_index=True)

st.markdown(
    f'<div style="border-top:1px solid {GRID};padding-top:.8rem;margin-top:1.5rem;'
    f'color:{MUTED};font-size:.75rem">NASA C-MAPSS FD001 · Linear Regression + StandardScaler pipeline · '
    f'SIH26054</div>',
    unsafe_allow_html=True,
)
