import streamlit as st
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import joblib

# Load the model and preprocessor
model = load_model('tf_bridge_model.h5')
scaler = joblib.load('scaler.pkl')

st.title("Bridge Load Capacity Prediction")

# User inputs
span_ft = st.number_input("Span (ft)", min_value=0)
deck = st.number_input("Deck Width (ft)", min_value=0)
age_years = st.number_input("Age (Years)", min_value=0)
num_lanes = st.number_input("Number of Lanes", min_value=0)
condition_rating = st.number_input("Condition Rating (1-5)", min_value=1, max_value=5)
material = st.selectbox("Material", options=["Steel", "Concrete", "Composite"])

# Preprocess inputs
input_data = pd.DataFrame({
    'Span_ft': [span_ft],
    'Deck_Width_ft': [deck],
    'Age_Years': [age_years],
    'Num_Lanes': [num_lanes],
    'Condition_Rating': [condition_rating],
    'Material_Steel': [1 if material == "Steel" else 0],
    'Material_Concrete': [1 if material == "Concrete" else 0],
    'Material_Composite': [1 if material == "Composite" else 0],
    'Placeholder': [0]  # Add a placeholder feature
})

numerical_features = ['Span_ft', 'Deck_Width_ft', 'Age_Years', 'Num_Lanes', 'Condition_Rating']
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Ensure input data shape matches the model's expected input shape
input_data = input_data.to_numpy().astype(np.float32)

# Print the model summary and input data shape
st.write(f"Model input shape: {model.input_shape}")
st.write(f"Input data shape: {input_data.shape}")

# Reshape input data if necessary
if len(input_data.shape) == 1:
    input_data = input_data.reshape(1, -1)

# Predict load capacity
prediction = model.predict(input_data)

st.write(f"Predicted Load Capacity: {prediction[0][0]}")
