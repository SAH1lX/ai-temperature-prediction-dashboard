import streamlit as st
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Temperature Dashboard",
    page_icon="🌡️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* ---------------- MAIN BACKGROUND ---------------- */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #334155);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* ---------------- SIDEBAR ---------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* ---------------- REMOVE STREAMLIT BRANDING ---------------- */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* ---------------- TOP PADDING ---------------- */
.block-container {
    padding-top: 1.5rem;
}

/* ---------------- CARD STYLE ---------------- */
.custom-card {
    background: rgba(30, 41, 59, 0.85);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.08);
}

/* ---------------- KPI CARDS ---------------- */
.kpi-card {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    color: white;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.3);
    transition: 0.3s;
}

.kpi-card:hover {
    transform: scale(1.03);
}

/* ---------------- METRIC TEXT ---------------- */
.metric-value {
    font-size: 30px;
    font-weight: bold;
}

.metric-label {
    font-size: 15px;
    opacity: 0.85;
}

/* ---------------- PREDICTION BOX ---------------- */
.prediction-box {
    background: linear-gradient(135deg, #14b8a6, #0f766e);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: white;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.3);
}

/* ---------------- WEATHER BOX ---------------- */
.weather-box {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: white;
    margin-top: 18px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.25);
}

/* ---------------- TITLES ---------------- */
h1, h2, h3 {
    color: white;
}

/* ---------------- INFO BOX ---------------- */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ---------------- CHART AREA ---------------- */
.chart-card {
    background: rgba(30, 41, 59, 0.88);
    padding: 15px;
    border-radius: 18px;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.25);
}

/* ---------------- FOOTER ---------------- */
.footer {
    text-align: center;
    padding: 20px;
    color: rgba(255,255,255,0.75);
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
title_col, time_col = st.columns([5,1])

with title_col:

    st.title("🌡️ AI Temperature Prediction Dashboard")

with time_col:

    now = datetime.now()

    st.markdown(f"""
    <div class="custom-card" style="text-align:center; padding:15px;">
        <h4>🕒 {now.strftime("%H:%M")}</h4>
        <p>{now.strftime("%d %b %Y")}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- DESCRIPTION ----------------
st.info(
    "Predict afternoon temperature using Machine Learning based on real weather conditions."
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Weather Controls")

humidity = st.sidebar.slider(
    "Humidity (%)",
    0,
    100,
    50
)

pressure = st.sidebar.slider(
    "Pressure (hPa)",
    980,
    1030,
    1010
)

wind = st.sidebar.slider(
    "Wind Speed (km/h)",
    0,
    50,
    10
)

temp9am = st.sidebar.slider(
    "Morning Temp (°C)",
    0,
    40,
    20
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Model Info")

st.sidebar.write("Algorithm: Linear Regression")
st.sidebar.write("Accuracy: 80%+")
st.sidebar.write("Version: 2.0")

# ---------------- MODEL PREDICTION ----------------
user = np.array([
    [humidity, pressure, wind, temp9am]
])

user_scaled = scaler.transform(user)

with st.spinner("Analyzing weather patterns..."):
    time.sleep(1)
    prediction = model.predict(user_scaled)[0]

# ---------------- WEATHER STATUS ----------------
if prediction < 15:

    weather = "❄️ Cold Weather"
    weather_color = "#2563eb"

elif prediction < 25:

    weather = "🌤️ Normal Weather"
    weather_color = "#f59e0b"

else:

    weather = "🔥 Hot Weather"
    weather_color = "#ef4444"

# ---------------- KPI SECTION ----------------
k1, k2, k3, k4 = st.columns(4)

with k1:

    st.markdown(f"""
    <div class="kpi-card">
        <div class="metric-value">{round(prediction,1)}°C</div>
        <div class="metric-label">Predicted Temp</div>
    </div>
    """, unsafe_allow_html=True)

with k2:

    st.markdown(f"""
    <div class="kpi-card">
        <div class="metric-value">{humidity}%</div>
        <div class="metric-label">Humidity</div>
    </div>
    """, unsafe_allow_html=True)

with k3:

    st.markdown(f"""
    <div class="kpi-card">
        <div class="metric-value">{wind} km/h</div>
        <div class="metric-label">Wind Speed</div>
    </div>
    """, unsafe_allow_html=True)

with k4:

    st.markdown(f"""
    <div class="kpi-card">
        <div class="metric-value">0.80+</div>
        <div class="metric-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- MAIN SECTION ----------------
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# ---------------- INPUT CARD ----------------
with col1:

    st.markdown("""
    <div class="custom-card">
    """, unsafe_allow_html=True)

    st.subheader("📥 Current Inputs")

    st.write(f"🌫️ Humidity: {humidity}%")
    st.write(f"📈 Pressure: {pressure} hPa")
    st.write(f"💨 Wind Speed: {wind} km/h")
    st.write(f"🌅 Morning Temp: {temp9am} °C")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION CARD ----------------
with col2:

    st.markdown("""
    <div class="custom-card">
    """, unsafe_allow_html=True)

    st.subheader("📌 Prediction Result")

    st.markdown(f"""
    <div class="prediction-box">
        🌡️ {round(prediction,2)} °C
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="weather-box"
         style="background:{weather_color};">
         {weather}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FEATURE VISUALIZATION ----------------
st.markdown("## 📊 Feature Visualization")

chart1, chart2 = st.columns(2)

features = [
    "Humidity",
    "Pressure",
    "Wind",
    "Temp9am"
]

values = [
    humidity,
    pressure,
    wind,
    temp9am
]

plt.style.use("dark_background")

# ---------------- BAR CHART ----------------
with chart1:

    st.markdown("""
    <div class="chart-card">
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6,4))

    bars = ax.bar(
        features,
        values,
        color=[
            "#38bdf8",
            "#ec4899",
            "#8b5cf6",
            "#22c55e"
        ]
    )

    ax.set_title("Input Features", fontsize=14)

    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")

    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{height}',
            ha='center',
            va='bottom',
            color='white'
        )

    st.pyplot(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HORIZONTAL CHART ----------------
with chart2:

    st.markdown("""
    <div class="chart-card">
    """, unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(6,4))

    ax2.barh(
        features,
        values,
        color="#38bdf8"
    )

    ax2.set_title("Feature Impact", fontsize=14)

    fig2.patch.set_facecolor("#1e293b")
    ax2.set_facecolor("#1e293b")

    st.pyplot(fig2, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- EXTRA INSIGHTS ----------------
st.markdown("## 🧠 AI Insights")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:

    st.markdown("""
    <div class="custom-card">
    """, unsafe_allow_html=True)

    st.subheader("📈 Weather Analysis")

    if humidity > 70:
        st.warning("High humidity detected. Possibility of rainfall.")

    elif humidity < 30:
        st.info("Low humidity. Dry weather conditions expected.")

    else:
        st.success("Humidity levels are moderate.")

    st.markdown("</div>", unsafe_allow_html=True)

with insight_col2:

    st.markdown("""
    <div class="custom-card">
    """, unsafe_allow_html=True)

    st.subheader("🤖 Model Recommendation")

    if prediction > 30:
        st.error("Very high temperature expected. Stay hydrated.")

    elif prediction < 15:
        st.warning("Cold weather expected. Wear warm clothes.")

    else:
        st.success("Weather conditions look comfortable today.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>

<div class="footer">

🚀 AI Weather Prediction Dashboard <br>

Developed by SAHIL BHATTI

</div>
""", unsafe_allow_html=True)