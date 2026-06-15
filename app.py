import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Geothermal Power Predictor", page_icon="🌋", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{font-size:2.1rem;font-weight:700;background:linear-gradient(90deg,#e65100,#ff9800);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.2rem;}
.sub-header{color:#666;font-size:0.95rem;margin-bottom:1rem;}
.result-card{background:linear-gradient(135deg,#e65100 0%,#bf360c 100%);border-radius:16px;padding:1.5rem;text-align:center;color:white;margin:1rem 0;}
.result-value{font-size:3rem;font-weight:800;color:#FFD54F;}
.result-label{font-size:1.05rem;opacity:0.9;margin-top:0.25rem;}
.info-box{background:#fff3e0;border-radius:8px;padding:0.8rem 1rem;border-left:4px solid #ff9800;font-size:0.88rem;color:#555;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    le = joblib.load("label_encoder.joblib")
    return model, le

@st.cache_resource
def geolocator():
    return Nominatim(user_agent="geothermal_predictor_app")

@st.cache_data(ttl=24*3600)
def reverse_country(lat, lon):
    try:
        loc = geolocator().reverse((lat, lon), language='en', zoom=3)
        if loc and loc.raw and 'address' in loc.raw:
            addr = loc.raw['address']
            return addr.get('country', 'Unknown')
    except Exception:
        pass
    return 'Unknown'

try:
    model, le = load_artifacts()
except Exception as e:
    st.error(f"Model artifacts not found: {e}")
    st.stop()

st.markdown('<div class="main-header">🌋 Geothermal Power Output Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict electrical power output from geothermal reservoir properties</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## About")
    st.markdown("This app predicts **power_generated_MWe** for production wells using a Random Forest model.")
    st.markdown("### Inputs")
    st.markdown("- Depth\n- Temperature\n- Pressure\n- Porosity\n- Permeability\n- Flow rate\n- Map location")
    st.markdown("### Formula reference")
    st.latex(r"P_{MWe} = \eta \cdot \dot{m} \cdot C_p \cdot \Delta T")

col_map, col_form = st.columns([1.2, 1])

with col_map:
    st.markdown("### 📍 Select location on map")
    m = folium.Map(location=[10, 110], zoom_start=2, tiles='OpenStreetMap')
    folium.LatLngPopup().add_to(m)
    out = st_folium(m, height=450, width=None, returned_objects=["last_clicked"])

    last_clicked = out.get("last_clicked") if out else None
    if last_clicked:
        lat = float(last_clicked['lat'])
        lon = float(last_clicked['lng'])
    else:
        lat = 7.12
        lon = 107.80

    country = reverse_country(lat, lon)

with col_form:
    st.markdown("### ⚙️ Well inputs")
    st.caption("Location is auto-filled from the map click")

    st.text_input("Detected Country", value=country, disabled=True)
    st.number_input("Latitude (°)", value=float(lat), format="%.4f", disabled=True)
    st.number_input("Longitude (°)", value=float(lon), format="%.4f", disabled=True)

    field_type = st.selectbox("Reservoir Type", ["Liquid-Dominated", "Vapor-Dominated", "EGS"], index=0)
    depth = st.slider("Well Depth (m)", 500, 6000, 2000, 50)
    temperature = st.slider("Temperature (°C)", 100, 450, 250, 5)
    pressure = st.slider("Pressure (MPa)", 1.0, 50.0, 18.0, 0.5)
    porosity = st.number_input("Porosity", 0.01, 0.25, 0.09, 0.005, format="%.4f")
    permeability = st.number_input("Permeability (mD)", 0.1, 2000.0, 45.0, 1.0, format="%.2f")
    flow_rate = st.number_input("Flow rate (kg/s)", 2.0, 120.0, 30.0, 0.5, format="%.2f")

predict = st.button("⚡ Predict Power Output", type="primary", use_container_width=True)

if predict:
    try:
        country_enc = int(le.transform([country])[0])
    except Exception:
        country_enc = 0

    row = pd.DataFrame([{
        'well_depth_m': depth,
        'temperature_C': temperature,
        'pressure_MPa': pressure,
        'porosity': porosity,
        'log_permeability': np.log1p(permeability),
        'log_flow_rate': np.log1p(flow_rate),
        'latitude': lat,
        'longitude': lon,
        'country_enc': country_enc,
        'field_type_EGS': 1 if field_type == 'EGS' else 0,
        'field_type_Liquid-Dominated': 1 if field_type == 'Liquid-Dominated' else 0,
        'field_type_Vapor-Dominated': 1 if field_type == 'Vapor-Dominated' else 0,
    }])

    pred = float(model.predict(row)[0])
    pred = max(pred, 0.0)
    physics_ref = max(0.0, flow_rate * 4.18 * max(temperature - 15, 0) * 0.12 / 1000)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-value">{pred:.3f}</div>
        <div class="result-label">Predicted Power (MWe)</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Physics reference", f"{physics_ref:.3f} MWe")
    c2.metric("Difference", f"{pred - physics_ref:+.3f} MWe")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pred,
        delta={"reference": physics_ref},
        gauge={"axis": {"range": [0, 12]}, "bar": {"color": "#e65100"}},
        title={"text": "Power Output Gauge"}
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download input row",
        data=row.to_csv(index=False),
        file_name="geothermal_input.csv",
        mime="text/csv"
    )
