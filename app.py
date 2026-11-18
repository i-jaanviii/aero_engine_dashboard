import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Aero Engine Dashboard", page_icon="✈️", layout="wide")

st.title("✈️ Aero Engine Predictive Maintenance")
st.subheader("SIH26054 | AI-Enabled Predictive Maintenance")

st.warning("DEMO MODE: All displayed values are simulated. These are not actual AI predictions.")

engine = st.selectbox("Select Engine", range(1, 101))
cycle = st.slider("Current Operating Cycle", 1, 300, 150)

rul = max(0, 300 - cycle)
status = "HEALTHY" if rul >= 100 else "WARNING" if rul >= 40 else "CRITICAL"

col1, col2, col3 = st.columns(3)
col1.metric("Engine Number", f"Engine {engine}")
col2.metric("Current Cycle", cycle)
col3.metric("Demo RUL", f"{rul} cycles")

if status == "HEALTHY":
    st.success(status)
elif status == "WARNING":
    st.warning(status)
else:
    st.error(status)

np.random.seed(engine)
df = pd.DataFrame({
    "Cycle": np.arange(1, 301),
    "Sensor 1": 500 + np.arange(300)*0.15 + np.random.normal(0, 3, 300),
    "Sensor 2": 600 - np.arange(300)*0.10 + np.random.normal(0, 4, 300),
    "Sensor 3": 20 + np.arange(300)*0.02 + np.random.normal(0, 0.5, 300),
    "Health": np.clip(100 - np.arange(300)*0.22, 0, 100)
})

st.header("Sensor Monitoring")
sensor = st.selectbox("Choose Sensor", ["Sensor 1", "Sensor 2", "Sensor 3"])
fig = px.line(df, x="Cycle", y=sensor)
st.plotly_chart(fig, use_container_width=True)

st.header("Historical Engine Degradation")
fig2 = px.line(df, x="Cycle", y="Health")
fig2.add_vline(x=cycle, line_dash="dash")
st.plotly_chart(fig2, use_container_width=True)

st.header("Model Performance Metrics")
a, b, c = st.columns(3)
a.metric("RMSE", "12.50")
b.metric("MAE", "8.40")
c.metric("Asymmetric Score", "150.20")
st.caption("All metrics are illustrative sample values.")
