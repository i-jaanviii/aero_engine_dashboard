
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aero Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#14213D"
BLUE = "#2563EB"
LIGHT_BLUE = "#60A5FA"
GREEN = "#16A34A"
ORANGE = "#F59E0B"
RED = "#DC2626"
BG = "#F4F7FB"

MODEL_PATH = "rul_model.joblib"
SENSOR_DATA_PATH = "dashboard_sensor_data.csv"
PREDICTIONS_PATH = "predictions.csv"


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F7FB;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #14213D;
        margin-bottom: 0px;
    }

    .sub-title {
        color: #64748B;
        font-size: 15px;
        margin-top: 3px;
    }

    .section-title {
        color: #14213D;
        font-size: 21px;
        font-weight: 700;
        margin-top: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 7px rgba(0,0,0,0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
    }

    div[data-testid="stMetricValue"] {
        color: #14213D;
        font-weight: 700;
    }

    .status-card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 12px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def calculate_nasa_score(actual, predicted):
    """
    NASA asymmetric scoring function.

    Negative error: prediction is lower than actual RUL.
    Positive error: prediction is higher than actual RUL.
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - actual

    scores = np.where(
        errors < 0,
        np.exp(-errors / 13) - 1,
        np.exp(errors / 10) - 1,
    )

    return float(np.sum(scores))


def normalize_column_names(data):
    """
    Standardizes common NASA C-MAPSS column naming formats.
    """

    data = data.copy()

    aliases = {
        "unit_nr": "engine_id",
        "unit_number": "engine_id",
        "time_cycles": "cycle",
        "operating_setting_1": "setting_1",
        "operating_setting_2": "setting_2",
        "operating_setting_3": "setting_3",
    }

    data = data.rename(columns=aliases)

    # Handle sensor columns such as sensor_02.
    rename = {}

    for col in data.columns:
        col_str = str(col).strip().lower()

        if col_str.startswith("sensor_"):
            suffix = col_str.replace("sensor_", "")

            try:
                sensor_number = int(suffix)
                rename[col] = f"sensor_{sensor_number}"
            except ValueError:
                pass

        elif col_str.startswith("s") and col_str[1:].isdigit():
            rename[col] = f"sensor_{int(col_str[1:])}"

    data = data.rename(columns=rename)

    return data


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        model = bundle.get("model")

        if model is None:
            raise ValueError(
                "The model bundle does not contain a 'model' key."
            )

        selected_sensors = bundle.get(
            "selected_sensors",
            [
                "sensor_2",
                "sensor_3",
                "sensor_4",
                "sensor_6",
                "sensor_7",
                "sensor_8",
                "sensor_9",
                "sensor_11",
                "sensor_12",
                "sensor_13",
                "sensor_14",
                "sensor_15",
                "sensor_17",
                "sensor_20",
                "sensor_21",
            ],
        )

        window_size = int(bundle.get("window_size", 30))
        rul_cap = float(bundle.get("rul_cap", 125))

    else:
        model = bundle

        selected_sensors = [
            "sensor_2",
            "sensor_3",
            "sensor_4",
            "sensor_6",
            "sensor_7",
            "sensor_8",
            "sensor_9",
            "sensor_11",
            "sensor_12",
            "sensor_13",
            "sensor_14",
            "sensor_15",
            "sensor_17",
            "sensor_20",
            "sensor_21",
        ]

        window_size = 30
        rul_cap = 125

    return model, selected_sensors, window_size, rul_cap


# ============================================================
# 5. LOAD SENSOR DATA
# ============================================================

@st.cache_data
def load_sensor_data():
    data = pd.read_csv(SENSOR_DATA_PATH)
    data = normalize_column_names(data)

    required_columns = {"engine_id", "cycle"}

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in sensor data: {sorted(missing)}"
        )

    data["engine_id"] = pd.to_numeric(
        data["engine_id"], errors="coerce"
    )

    data["cycle"] = pd.to_numeric(
        data["cycle"], errors="coerce"
    )

    data = data.dropna(subset=["engine_id", "cycle"])

    data["engine_id"] = data["engine_id"].astype(int)
    data["cycle"] = data["cycle"].astype(int)

    data = data.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    return data


# ============================================================
# 6. LOAD COLAB EVALUATION RESULTS
# ============================================================

@st.cache_data
def load_evaluation_results():

    results = pd.read_csv(PREDICTIONS_PATH)

    required = {
        "engine_id",
        "predicted_RUL",
        "actual_RUL",
    }

    missing = required.difference(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in predictions.csv: {sorted(missing)}"
        )

    results["engine_id"] = pd.to_numeric(
        results["engine_id"], errors="coerce"
    )

    results["predicted_RUL"] = pd.to_numeric(
        results["predicted_RUL"], errors="coerce"
    )

    results["actual_RUL"] = pd.to_numeric(
        results["actual_RUL"], errors="coerce"
    )

    results = results.dropna(
        subset=["engine_id", "predicted_RUL", "actual_RUL"]
    )

    results = results.sort_values(
        "engine_id"
    ).reset_index(drop=True)

    if len(results) != 100:
        raise ValueError(
            f"Expected 100 test engine predictions, found {len(results)}."
        )

    if results["engine_id"].nunique() != 100:
        raise ValueError(
            "predictions.csv must contain 100 unique engine IDs."
        )

    actual = results["actual_RUL"].to_numpy(dtype=float)
    predicted = results["predicted_RUL"].to_numpy(dtype=float)

    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    mae = float(np.mean(np.abs(predicted - actual)))
    score = calculate_nasa_score(actual, predicted)

    return rmse, mae, score


# ============================================================
# 7. PREDICT RUL FOR SELECTED ENGINE
# ============================================================

def predict_engine_rul(
    engine_data,
    model,
    selected_sensors,
    window_size,
    rul_cap,
    selected_cycle=None,
):

    engine_data = engine_data.sort_values("cycle").copy()

    if selected_cycle is not None:
        engine_data = engine_data[
            engine_data["cycle"] <= selected_cycle
        ]

    if len(engine_data) < window_size:
        raise ValueError(
            f"At least {window_size} cycles are required for prediction."
        )

    missing_sensors = [
        sensor for sensor in selected_sensors
        if sensor not in engine_data.columns
    ]

    if missing_sensors:
        raise ValueError(
            f"Missing model input sensors: {missing_sensors}"
        )

    latest_window = engine_data.tail(window_size)

    sensor_values = latest_window[
        selected_sensors
    ].apply(pd.to_numeric, errors="coerce")

    if sensor_values.isna().any().any():
        raise ValueError(
            "Sensor data contains missing or non-numeric values."
        )

    # 30 cycles x 15 sensors = 450 features.
    features = sensor_values.to_numpy().reshape(1, -1)

    # The saved pipeline already includes StandardScaler.
    prediction = model.predict(features)[0]

    prediction = float(np.clip(prediction, 0, rul_cap))

    return prediction


# ============================================================
# 8. HISTORICAL DEGRADATION PREDICTIONS
# ============================================================

def generate_degradation_history(
    engine_data,
    model,
    selected_sensors,
    window_size,
    rul_cap,
):

    engine_data = engine_data.sort_values("cycle").copy()

    history = []

    for end_index in range(window_size, len(engine_data) + 1):

        window = engine_data.iloc[
            end_index - window_size:end_index
        ]

        sensor_values = window[
            selected_sensors
        ].apply(pd.to_numeric, errors="coerce")

        if sensor_values.isna().any().any():
            continue

        features = sensor_values.to_numpy().reshape(1, -1)

        prediction = float(
            model.predict(features)[0]
        )

        prediction = float(
            np.clip(prediction, 0, rul_cap)
        )

        history.append({
            "cycle": int(window.iloc[-1]["cycle"]),
            "predicted_RUL": prediction,
        })

    return pd.DataFrame(history)


# ============================================================
# 9. HEALTH STATUS
# ============================================================

def get_health_status(rul):

    if rul > 50:
        return "Healthy", GREEN

    elif rul > 20:
        return "Warning", ORANGE

    else:
        return "Critical", RED


# ============================================================
# 10. MAIN DASHBOARD
# ============================================================

st.markdown(
    '<div class="main-title">✈️ Aero Engine Predictive Maintenance</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'SIH26054 | AI-Enabled Predictive Maintenance for Aero Engines'
    '</div>',
    unsafe_allow_html=True,
)

st.write("")

st.info(
    "This dashboard uses the trained NASA C-MAPSS FD001 model "
    "to estimate Remaining Useful Life (RUL) from engine sensor readings."
)


# ============================================================
# 11. LOAD PROJECT FILES
# ============================================================

try:
    model, selected_sensors, window_size, rul_cap = load_model()
    sensor_data = load_sensor_data()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# ============================================================
# 12. SIDEBAR - ENGINE SELECTION
# ============================================================

st.sidebar.title("Dashboard Controls")

st.sidebar.markdown("---")

engine_ids = sorted(
    sensor_data["engine_id"].unique().tolist()
)

if not engine_ids:
    st.error("No engine data found in dashboard_sensor_data.csv.")
    st.stop()

selected_engine = st.sidebar.selectbox(
    "Select Engine",
    engine_ids,
    index=0,
)

engine_data = sensor_data[
    sensor_data["engine_id"] == selected_engine
].sort_values("cycle").copy()

if len(engine_data) < window_size:
    st.error(
        f"Engine {selected_engine} has fewer than "
        f"{window_size} cycles. Prediction is not possible."
    )
    st.stop()

minimum_cycle = int(engine_data["cycle"].iloc[window_size - 1])
maximum_cycle = int(engine_data["cycle"].max())

selected_cycle = st.sidebar.slider(
    "Operating Cycle",
    min_value=minimum_cycle,
    max_value=maximum_cycle,
    value=maximum_cycle,
    step=1,
)

st.sidebar.markdown("---")

st.sidebar.caption(
    f"Selected Engine: {selected_engine}"
)

st.sidebar.caption(
    f"Available Cycles: {len(engine_data)}"
)

st.sidebar.caption(
    f"Model Window: {window_size} cycles"
)

# ============================================================
# 13. LIVE RUL PREDICTION
# ============================================================

try:
    predicted_rul = predict_engine_rul(
        engine_data=engine_data,
        model=model,
        selected_sensors=selected_sensors,
        window_size=window_size,
        rul_cap=rul_cap,
        selected_cycle=selected_cycle,
    )

except Exception as error:
    st.error(f"RUL prediction failed: {error}")
    st.stop()


health_status, status_color = get_health_status(predicted_rul)


# ============================================================
# 14. TOP METRIC CARDS
# ============================================================

st.markdown(
    '<div class="section-title">Engine Overview</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Engine ID",
        f"Engine {selected_engine}",
    )

with col2:
    st.metric(
        "Operating Cycle",
        selected_cycle,
    )

with col3:
    st.metric(
        "Predicted Remaining Useful Life",
        f"{predicted_rul:.2f} cycles",
    )

with col4:
    st.metric(
        "Health Status",
        health_status,
    )


# ============================================================
# 15. ENGINE HEALTH INDICATOR
# ============================================================

st.markdown(
    '<div class="section-title">Engine Health Status</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="status-card">
        <h3 style="color:{status_color}; margin:0;">
            ● {health_status}
        </h3>
        <p style="color:#64748B; margin-top:8px;">
            Estimated remaining useful life:
            <b>{predicted_rul:.2f} cycles</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.progress(
    min(max(predicted_rul / rul_cap, 0.0), 1.0),
    text=f"Remaining life indicator: {predicted_rul:.1f} / {rul_cap:.0f}",
)


# ============================================================
# 16. SENSOR MONITORING GRAPHS
# ============================================================

st.markdown(
    '<div class="section-title">Sensor Monitoring</div>',
    unsafe_allow_html=True,
)

sensor_window = engine_data[
    engine_data["cycle"] <= selected_cycle
].tail(window_size).copy()

available_sensors = [
    sensor for sensor in selected_sensors
    if sensor in sensor_window.columns
]

# Display the first six selected model sensors.
display_sensors = available_sensors[:6]

if display_sensors:

    fig_sensors = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=display_sensors,
        vertical_spacing=0.13,
        horizontal_spacing=0.10,
    )

    for index, sensor in enumerate(display_sensors):

        row = index // 2 + 1
        col = index % 2 + 1

        fig_sensors.add_trace(
            go.Scatter(
                x=sensor_window["cycle"],
                y=sensor_window[sensor],
                mode="lines+markers",
                name=sensor,
                line=dict(width=2, color=BLUE),
                marker=dict(size=4),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig_sensors.update_xaxes(
            title_text="Operating Cycle",
            row=row,
            col=col,
        )

        fig_sensors.update_yaxes(
            title_text="Sensor Reading",
            row=row,
            col=col,
        )

    fig_sensors.update_layout(
        height=650,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(
        fig_sensors,
        use_container_width=True,
    )

else:
    st.warning("No selected sensor columns are available to plot.")


# ============================================================
# 17. HISTORICAL DEGRADATION CURVE
# ============================================================

st.markdown(
    '<div class="section-title">Historical RUL Degradation Curve</div>',
    unsafe_allow_html=True,
)

try:
    degradation_history = generate_degradation_history(
        engine_data=engine_data[
            engine_data["cycle"] <= selected_cycle
        ],
        model=model,
        selected_sensors=selected_sensors,
        window_size=window_size,
        rul_cap=rul_cap,
    )

    if not degradation_history.empty:

        fig_degradation = go.Figure()

        fig_degradation.add_trace(
            go.Scatter(
                x=degradation_history["cycle"],
                y=degradation_history["predicted_RUL"],
                mode="lines+markers",
                name="Predicted RUL",
                line=dict(
                    color=BLUE,
                    width=3,
                ),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.08)",
            )
        )

        fig_degradation.add_trace(
            go.Scatter(
                x=[selected_cycle],
                y=[predicted_rul],
                mode="markers",
                name="Current Prediction",
                marker=dict(
                    color=RED,
                    size=12,
                    symbol="diamond",
                ),
            )
        )

        fig_degradation.update_layout(
            title=f"Engine {selected_engine} - RUL over Operating Cycles",
            xaxis_title="Operating Cycle",
            yaxis_title="Predicted Remaining Useful Life",
            template="plotly_white",
            height=420,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=60, b=20),
        )

        st.plotly_chart(
            fig_degradation,
            use_container_width=True,
        )

    else:
        st.warning(
            "Not enough valid sensor readings to generate the degradation curve."
        )

except Exception as error:
    st.warning(
        f"Unable to generate degradation curve: {error}"
    )


# ============================================================
# 18. MODEL PERFORMANCE METRICS
# ============================================================

st.markdown(
    '<div class="section-title">Model Performance Evaluation</div>',
    unsafe_allow_html=True,
)

try:
    rmse, mae, asymmetric_score = load_evaluation_results()

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "RMSE",
            f"{rmse:.3f}",
        )

    with metric2:
        st.metric(
            "MAE",
            f"{mae:.3f}",
        )

    with metric3:
        st.metric(
            "NASA Asymmetric Score",
            f"{asymmetric_score:.3f}",
        )

    st.caption(
        "Metrics are calculated from the saved predictions.csv "
        "file containing predictions and ground-truth RUL labels "
        "for the 100 FD001 test engines. These are evaluation "
        "results, not live metrics for the selected engine."
    )

except Exception as error:
    st.error(
        f"Unable to load evaluation metrics: {error}"
    )


# ============================================================
# 19. SELECTED ENGINE DATA TABLE
# ============================================================

with st.expander("View Selected Engine Sensor Data"):

    st.dataframe(
        sensor_window[
            ["engine_id", "cycle"] + available_sensors
        ],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 20. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        SIH26054 | AI-Enabled Predictive Maintenance for Aero Engines
        <br>
        Dataset: NASA C-MAPSS FD001
        <br>
        Model: Linear Regression
        <br>
        Dashboard: Python | Streamlit | Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
