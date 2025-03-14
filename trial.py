import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained RandomForest model
model = joblib.load("random_forest_pkl.pkl")  # Replace with your model path

# Streamlit UI
st.title("Soil Fertility Prediction")

st.markdown("### Enter the soil parameters below:")

# Input fields for the 12 features
N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=1000.0, step=0.1)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=1000.0, step=0.1)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=1000.0, step=0.1)
pH = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
EC = st.number_input("Electrical Conductivity (EC)", min_value=0.0, max_value=5.0, step=0.01)
OC = st.number_input("Organic Carbon (OC)", min_value=0.0, max_value=10.0, step=0.01)
S = st.number_input("Sulfur (S)", min_value=0.0, max_value=500.0, step=0.1)
Zn = st.number_input("Zinc (Zn)", min_value=0.0, max_value=100.0, step=0.01)
Fe = st.number_input("Iron (Fe)", min_value=0.0, max_value=500.0, step=0.01)
Cu = st.number_input("Copper (Cu)", min_value=0.0, max_value=100.0, step=0.01)
Mn = st.number_input("Manganese (Mn)", min_value=0.0, max_value=100.0, step=0.01)
B = st.number_input("Boron (B)", min_value=0.0, max_value=100.0, step=0.01)

# Prediction button
if st.button("Predict Fertility"):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame([[N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B]], 
                              columns=['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Map prediction to fertility level
    fertility_mapping = {0: "Infertile", 1: "Fertile", 2: "Heavy Fertile"}
    result = fertility_mapping.get(prediction, "Unknown")
    
    # Display prediction
    st.success(f"Predicted Soil Fertility: {result}")
