import streamlit as st
import requests
import pandas as pd
import json
import mlflow
from datetime import datetime

# Set MLflow tracking URI and define experiment for monitoring predictions
mlflow.set_tracking_uri("file:///Users/touka/Desktop/BAU/forth year/s2/AIN3009/project/Mlflow_project/mlruns")
mlflow.set_experiment("Diabetes_Monitoring")

# Streamlit app title
st.title(" 🍬🩺 Diabetes Prediction App")

# Input form for prediction
with st.form("prediction_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

    submitted = st.form_submit_button("Predict") # Submit button

# # When user submits the form send data to deployed MLflow model
if submitted:
    input_data = {
        "dataframe_split": {
            "columns": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
            "data": [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age
            ]]
        }
    }

    # Send POST request to local MLflow model server
    response = requests.post(
        url="http://127.0.0.1:1234/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(input_data)
    )
    # prediction response
    if response.status_code == 200:
        prediction = response.json()["predictions"][0]
        label = "Diabetic" if prediction == 1 else "Not_Diabetic"
        st.success(f" Prediction: {label.replace('_', ' ')}")

        # Log to MLflow
        with mlflow.start_run(run_name="User Prediction"):
            mlflow.set_tag("source", "streamlit_app")
            mlflow.set_tag("timestamp", datetime.now().isoformat())
            mlflow.set_tag("label", label)

            # Log input values as parameters
            mlflow.log_param("Pregnancies", pregnancies)
            mlflow.log_param("Glucose", glucose)
            mlflow.log_param("BloodPressure", blood_pressure)
            mlflow.log_param("SkinThickness", skin_thickness)
            mlflow.log_param("Insulin", insulin)
            mlflow.log_param("BMI", bmi)
            mlflow.log_param("DiabetesPedigreeFunction", dpf)
            mlflow.log_param("Age", age)

            # Log prediction class (0 or 1)
            mlflow.log_metric("prediction", prediction)

    else:
        st.error(f"Error: {response.text}")

