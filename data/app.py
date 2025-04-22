import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Renewable Energy Forecast", layout="centered")

st.title("Renewable Energy Production Forecasting System")
st.markdown("Predict solar **radiation** (energy output) using weather data.")

@st.cache_data
def load_data():
    df = pd.read_csv("SolarPrediction.csv")
    return df

data = load_data()

features = ['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed']
target = 'Radiation'
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.sidebar.header("Input Weather Features")
temperature = st.sidebar.slider("Temperature (°C)", float(data['Temperature'].min()), float(data['Temperature'].max()), 25.0)
pressure = st.sidebar.slider("Pressure (mbar)", float(data['Pressure'].min()), float(data['Pressure'].max()), 1010.0)
humidity = st.sidebar.slider("Humidity (%)", float(data['Humidity'].min()), float(data['Humidity'].max()), 50.0)
wind_dir = st.sidebar.slider("Wind Direction (°)", float(data['WindDirection(Degrees)'].min()), float(data['WindDirection(Degrees)'].max()), 180.0)
speed = st.sidebar.slider("Wind Speed (m/s)", float(data['Speed'].min()), float(data['Speed'].max()), 2.0)

input_data = pd.DataFrame({
    'Temperature': [temperature],
    'Pressure': [pressure],
    'Humidity': [humidity],
    'WindDirection(Degrees)': [wind_dir],
    'Speed': [speed]
})

if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Solar Radiation: **{prediction:.2f} kW/m²**")

st.subheader("Model Performance")
st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**RMSE:** {rmse:.4f}")

st.subheader("Actual vs Predicted Radiation")
fig, ax = plt.subplots()
ax.plot(y_test.values[:100], label="Actual", marker='o', linestyle='--')
ax.plot(y_pred[:100], label="Predicted", marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Radiation (kW/m²)")
plt.legend()
plt.grid(True)
st.pyplot(fig)
