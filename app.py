
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
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#14213D"
BLUE = "#38BDF8"
GREEN = "#22C55E"
ORANGE = "#F59E0B"
RED = "#F43F5E"

MODEL_PATH = "rul_model.joblib"
SENSOR_DATA_PATH = "dashboard_sensor_data.csv"
PREDICTIONS_PATH = "predictions.csv"


# ============================================================
# 2. DARK NAVY THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B1728;
        color: #EAF2FF;
    }

    [data-testid="stSidebar"] {
        background-color: #07111F;
    }

    [data-testid="stHeader"] {
        background-color: #0B1728;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #F1F5F9;
        margin-bottom: 0;
    }

    .sub-title {
        color: #93C5FD;
        font-size: 15px;
        margin-top: 8px;
    }

    .section-title {
        color: #F1F5F9;
        font-size: 23px;
        font-weight: 700;
        margin-top: 20px;
    }

    div[data-testid="stMetric"] {
        background-color: #112238;
        border: 1px solid #263B53;
        padding: 18px;
        border-radius: 12px;
        box-shadow: none;
    }

    div[data-testid="stMetricLabel"] {
        color: #93C5FD;
    }

    div[data-testid="stMetricValue"] {
        color: #F1F5F9;
        font-weight: 700;
    }

    .status-card {
        background-color: #112238;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #263B53;
        margin-bottom: 10px;
    }

    .footer {
        text-align: center;
        color: #94A3B8;
        font-size: 12px;
        padding: 20px;
    }

    div[data-baseweb="select"] > div {
        background-color: #112238;
        color: #F1F5F9;
        border-color: #344B65;
    }

    div[data-testid="stAlert"] {
        background-color: #112238;
        color: #EAF2FF;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def calculate_nasa_score(actual, predicted):

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

    rename = {}

    for col in data.columns:

        col_str = str(col).strip().lower()

        if col_str.startswith("sensor_"):

            suffix = col_str.replace("sensor_", "")

            try:
                rename[col] = f"sensor_{int(suffix)}"
            except ValueError:
                pass

        elif col_str.startswith("s") and col_str[1:].isdigit():

            rename[col] = f"sensor_{int(col_str[1:])}"

    data = data.rename(columns=rename)

    return data


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    bundle = joblib.load(MODEL_PATH)

    default_sensors = [
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

    if isinstance(bundle, dict):

        model = bundle.get("model")

        if model is None:
            raise ValueError(
                "Model bundle does not contain a 'model' key."
            )

        selected_sensors = bundle.get(
            "selected_sensors",
            default_sensors,
        )

        window_size = int(bundle.get("window_size", 30))
        rul_cap = float(bundle.get("rul_cap", 125))

    else:

        model = bundle
        selected_sensors = default_sensors
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

    required = {"engine_id", "cycle"}

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns in sensor data: {sorted(missing)}"
        )

    data["engine_id"] = pd.to_numeric(
        data["engine_id"], errors="coerce"
    )

    data["cycle"] = pd.to_numeric(
        data["cycle"], errors="coerce"
    )

    data = data.dropna(
        subset=["engine_id", "cycle"]
    )

    data["engine_id"] = data["engine_id"].astype(int)
    data["cycle"] = data["cycle"].astype(int)

    data = data.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    return data


# ============================================================
# 6. LOAD MODEL EVALUATION RESULTS
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

    for col in required:

        results[col] = pd.to_numeric(
            results[col], errors="coerce"
        )

    results = results.dropna(
        subset=list(required)
    )

    results = results.sort_values(
        "engine_id"
    ).reset_index(drop=True)

    if len(results) != 100:
        raise ValueError(
            f"Expected 100 engine predictions, found {len(results)}."
        )

    if results["engine_id"].nunique() != 100:
        raise ValueError(
            "predictions.csv must contain 100 unique engine IDs."
        )

    actual = results["actual_RUL"].to_numpy(dtype=float)

    predicted = results["predicted_RUL"].to_numpy(dtype=float)

    rmse = float(
        np.sqrt(np.mean((predicted - actual) ** 2))
    )

    mae = float(
        np.mean(np.abs(predicted - actual))
    )

    score = calculate_nasa_score(actual, predicted)

    return rmse, mae, score


# ============================================================
# 7. PREDICT RUL
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
            f"At least {window_size} cycles are required."
        )

    missing_sensors = [
        sensor for sensor in selected_sensors
        if sensor not in engine_data.columns
    ]

    if missing_sensors:

        raise ValueError(
            f"Missing sensor columns: {missing_sensors}"
        )

    latest_window = engine_data.tail(window_size)

    sensor_values = latest_window[
        selected_sensors
    ].apply(pd.to_numeric, errors="coerce")

    if sensor_values.isna().any().any():

        raise ValueError(
            "Sensor data contains missing or non-numeric values."
        )

    features = sensor_values.to_numpy().reshape(1, -1)

    # Scaling is already included in the trained pipeline.
    prediction = model.predict(features)[0]

    return float(
        np.clip(prediction, 0, rul_cap)
    )


# ============================================================
# 8. HISTORICAL RUL PREDICTIONS
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

    for end_index in range(
        window_size,
        len(engine_data) + 1
    ):

        window = engine_data.iloc[
            end_index - window_size:end_index
        ]

        sensor_values = window[
            selected_sensors
        ].apply(pd.to_numeric, errors="coerce")

        if sensor_values.isna().any().any():
            continue

        features = sensor_values.to_numpy().reshape(1, -1)

        prediction = model.predict(features)[0]

        prediction = float(
            np.clip(prediction, 0, rul_cap)
        )

        history.append({
            "cycle": int(window.iloc[-1]["cycle"]),
            "predicted_RUL": prediction,
        })

    return pd.DataFrame(history)


# ============================================================
# 9. ENGINE HEALTH STATUS
# ============================================================

def get_health_status(rul):

    if rul > 50:
        return "Healthy", GREEN

    elif rul > 20:
        return "Warning", ORANGE

    else:
        return "Critical", RED


# ============================================================
# 10. DASHBOARD HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Aero Engine Predictive Maintenance</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
    NASA C-MAPSS FD001 | Engine condition monitoring |
    Remaining useful life prediction
    </div>
    """,
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
# 12. SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("Monitoring controls")

st.sidebar.markdown("---")

engine_ids = sorted(
    sensor_data["engine_id"].unique().tolist()
)

if not engine_ids:

    st.error("No engine data found.")

    st.stop()

selected_engine = st.sidebar.selectbox(
    "Select engine",
    engine_ids,
    format_func=lambda x: f"Engine {x}",
)

engine_data = sensor_data[
    sensor_data["engine_id"] == selected_engine
].sort_values("cycle").copy()

if len(engine_data) < window_size:

    st.error(
        f"Engine {selected_engine} has fewer than "
        f"{window_size} cycles."
    )

    st.stop()

minimum_cycle = int(
    engine_data["cycle"].iloc[window_size - 1]
)

maximum_cycle = int(
    engine_data["cycle"].max()
)

selected_cycle = st.sidebar.slider(
    "Current operating cycle",
    min_value=minimum_cycle,
    max_value=maximum_cycle,
    value=maximum_cycle,
    step=1,
)

st.sidebar.markdown("---")

st.sidebar.caption("Dataset: NASA C-MAPSS FD001")

st.sidebar.caption("Model: Linear Regression")

st.sidebar.caption(
    f"Input: {window_size} cycles × {len(selected_sensors)} sensors"
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


health_status, status_color = get_health_status(
    predicted_rul
)


# ============================================================
# 14. ENGINE OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">Engine overview</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Selected engine",
        f"ENG-{selected_engine:03d}",
    )

with col2:

    st.metric(
        "Current operating cycle",
        selected_cycle,
    )

with col3:

    st.metric(
        "Predicted remaining useful life",
        f"{predicted_rul:.2f} cycles",
    )

with col4:

    st.metric(
        "Health status (RUL threshold rule)",
        health_status,
    )


# ============================================================
# 15. HEALTH STATUS INDICATOR
# ============================================================

st.markdown(
    f"""
    <div class="status-card">
        <h3 style="color:{status_color}; margin:0;">
            {health_status}
        </h3>
        <p style="color:#94A3B8; margin-top:8px;">
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
# 16. SENSOR MONITORING
# ============================================================

st.markdown(
    '<div class="section-title">Sensor monitoring</div>',
    unsafe_allow_html=True,
)

sensor_window = engine_data[
    engine_data["cycle"] <= selected_cycle
].tail(window_size).copy()

available_sensors = [
    sensor for sensor in selected_sensors
    if sensor in sensor_window.columns
]

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
                line=dict(width=2, color="#38BDF8"),
                marker=dict(size=4),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig_sensors.update_xaxes(
            title_text="Operating cycle",
            row=row,
            col=col,
        )

        fig_sensors.update_yaxes(
            title_text="Sensor value",
            row=row,
            col=col,
        )

    fig_sensors.update_layout(
        height=650,
        template="plotly_dark",
        paper_bgcolor="#112238",
        plot_bgcolor="#112238",
        font=dict(color="#EAF2FF"),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(
        fig_sensors,
        use_container_width=True,
    )

else:

    st.warning("No selected sensor columns are available.")


# ============================================================
# 17. HISTORICAL RUL PREDICTION CURVE
# ============================================================

st.markdown(
    '<div class="section-title">Historical RUL prediction curve</div>',
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
                    color="#A78BFA",
                    width=3,
                ),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(167,139,250,0.08)",
            )
        )

        fig_degradation.add_trace(
            go.Scatter(
                x=[selected_cycle],
                y=[predicted_rul],
                mode="markers",
                name="Current prediction",
                marker=dict(
                    color="#F43F5E",
                    size=12,
                    symbol="diamond",
                ),
            )
        )

        fig_degradation.update_layout(
            title="Predicted RUL over historical windows",
            xaxis_title="Operating cycle",
            yaxis_title="Predicted remaining life (cycles)",
            template="plotly_dark",
            paper_bgcolor="#112238",
            plot_bgcolor="#112238",
            font=dict(color="#EAF2FF"),
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
            "Not enough valid sensor readings for the degradation curve."
        )

except Exception as error:

    st.warning(
        f"Unable to generate degradation curve: {error}"
    )


# ============================================================
# 18. MODEL PERFORMANCE METRICS
# ============================================================

st.markdown(
    '<div class="section-title">Model performance evaluation</div>',
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
            "NASA asymmetric score",
            f"{asymmetric_score:.3f}",
        )

    st.caption(
        "Metrics are calculated from predictions.csv, containing "
        "predictions and ground-truth RUL labels for 100 FD001 test "
        "engines. These are evaluation results, not live metrics "
        "for the selected engine."
    )

except Exception as error:

    st.error(
        f"Unable to load evaluation metrics: {error}"
    )


# ============================================================
# 19. SENSOR DATA TABLE
# ============================================================

with st.expander("View selected engine sensor data"):

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
