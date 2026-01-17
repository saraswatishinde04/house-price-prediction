import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("models/house_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction App")
st.write("Enter house details to predict the median house price.")

# Input fields
med_inc = st.number_input("Median Income", min_value=0.0, value=3.0)
house_age = st.number_input("House Age", min_value=0.0, value=20.0)
avg_rooms = st.number_input("Average Rooms", min_value=0.0, value=5.0)
avg_bedrooms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0)
population = st.number_input("Population", min_value=0.0, value=1000.0)
avg_occupancy = st.number_input("Average Occupancy", min_value=0.0, value=3.0)
latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0)
longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-118.0)

if st.button("Predict Price"):
    features = np.array([[ 
        med_inc,
        house_age,
        avg_rooms,
        avg_bedrooms,
        population,
        avg_occupancy,
        latitude,
        longitude
    ]])

    # Scale input
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    st.success(f"🏷️ Predicted Median House Price: ${prediction * 100000:.2f}")
