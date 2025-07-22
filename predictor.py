import gradio as gr
import numpy as np
import joblib
import os
from keras.models import load_model

# Load scaler
scaler = joblib.load("saved_models/scaler.pkl")

# Load model
model = None
model_type = None
if os.path.exists("saved_models/best_model.h5"):
    model = load_model("saved_models/best_model.h5")
    model_type = "keras"
elif os.path.exists("saved_models/best_model.pkl"):
    model = joblib.load("saved_models/best_model.pkl")
    model_type = "sklearn"
else:
    raise Exception("No model found in saved_models directory.")

# Prediction function
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal):
    
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    input_scaled = scaler.transform(input_data)
    
    if model_type == "keras":
        prediction = model.predict(input_scaled)
        prediction = (prediction > 0.5).astype(int)
    else:
        prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        return " High risk of heart attack"
    else:
        return " No risk detected"

# Interface
inputs = [
    gr.Number(label="Age", value=50),
    gr.Radio([0, 1], label="Sex (0=Female, 1=Male)"),
    gr.Radio([0, 1, 2, 3], label="Chest Pain Type"),
    gr.Number(label="Resting Blood Pressure (mm Hg)", value=120),
    gr.Number(label="Serum Cholesterol (mg/dl)", value=200),
    gr.Radio([0, 1], label="Fasting Blood Sugar > 120"),
    gr.Radio([0, 1, 2], label="Resting ECG Results"),
    gr.Number(label="Max Heart Rate Achieved", value=150),
    gr.Radio([0, 1], label="Exercise Induced Angina"),
    gr.Number(label="ST Depression (oldpeak)", value=1.0),
    gr.Radio([0, 1, 2], label="Slope of ST Segment"),
    gr.Radio([0, 1, 2, 3], label="Number of Major Vessels"),
    gr.Radio([1, 2, 3], label="Thalassemia")
]

demo = gr.Interface(
    fn=predict_heart_disease,
    inputs=inputs,
    outputs="text",
    title=" Heart Attack Prediction",
    description="Enter your data to calculate risk."
)

demo.launch()
# It launches the Gradio interface on a local URL which is given in the terminal, the public one expires.