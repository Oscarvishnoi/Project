
import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load("shopper_model.joblib")

st.title("🛒 Shopper Purchase Prediction")

st.markdown("Enter the session details to predict if the shopper will make a purchase.")

# Input fields
features = []

feature_names = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month (1=Jan,...,12=Dec)", "OperatingSystems", "Browser",
    "Region", "TrafficType", 
    "VisitorType (1=Returning, 0=New/Other)", 
    "Weekend (1=Yes, 0=No)"
]

defaults = [2, 60.0, 0, 0.0, 20, 600.0, 0.02, 0.05, 10.0, 0.0, 6, 2, 1, 3, 2, 1, 0]

for name, default in zip(feature_names, defaults):
    value = st.number_input(name, value=default, format="%.4f" if isinstance(default, float) else "%d")
    features.append(value)

# Prediction
if st.button("Predict"):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    result = "🟢 Yes, the shopper will likely purchase!" if prediction else "🔴 No, the shopper is unlikely to purchase."
    st.success(result)
