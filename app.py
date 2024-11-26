
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("decision_tree_regressor.pkl")

# Title
st.title("Sleep Disorder Prediction")

# Sidebar for input
st.sidebar.header("Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 80, 30)
    quality_of_sleep = st.sidebar.slider("Quality of Sleep", 1, 10, 5)
    physical_activity_level = st.sidebar.slider("Physical Activity Level", 1, 10, 5)
    stress_level = st.sidebar.slider("Stress Level", 1, 10, 5)
    heart_rate = st.sidebar.slider("Heart Rate", 50, 100, 70)
    daily_steps = st.sidebar.slider("Daily Steps", 1000, 20000, 5000)
    bmi_category = st.sidebar.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
    
    data = {
        "Age": age,
        "Quality of Sleep": quality_of_sleep,
        "Physical Activity Level": physical_activity_level,
        "Stress Level": stress_level,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps,
        "BMI Category": bmi_category,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display user inputs
st.subheader("User Input Parameters")
st.write(input_df)

# Encode categorical data
bmi_mapping = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
input_df['BMI Category'] = input_df['BMI Category'].map(bmi_mapping)

# Predict using the model
prediction = model.predict(input_df)

# Display the prediction
st.subheader("Prediction")
st.write(f"Predicted Sleep Duration: {prediction[0]:.2f} hours")
