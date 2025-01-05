import os
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Load pre-trained model, scaler, and imputer
MODEL_PATH = 'model/pcos_svm_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
IMPUTER_PATH = 'model/imputer.pkl'

# Check if model, scaler, and imputer exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(IMPUTER_PATH):
    raise FileNotFoundError("Model, scaler, or imputer file is missing! Please ensure all necessary files are in the 'model' folder.")

# Load the model, scaler, and imputer
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
imputer = joblib.load(IMPUTER_PATH)

# Define the feature names as in the dataset
FEATURE_NAMES = [
    "Age (yrs)", "Weight (Kg)", "Height(Cm)", "BMI", "Blood Group", "Cycle(R/I)", "Cycle length(days)",
    "Marriage Status (Yrs)", "Pregnant(Y/N)", "No. of aborptions", "Weight gain(Y/N)", "hair growth(Y/N)",
    "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)"
]

# Define Blood Group mapping
blood_group_mapping = {
    "A+": 11, "A-": 12, "B+": 13, "B-": 14, "O+": 15, "O-": 16, "AB+": 17, "AB-": 18
}

# Define Y/N mapping for categorical features
yn_mapping = {
    "Y": 1, "N": 0
}

# Route for the home page
@app.route('/')
def home():
    return "PCOS Prediction Model API is running!"


# Route to predict PCOS based on user input
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure request contains JSON data
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Ensure all features are included in the input data
    if len(data) != len(FEATURE_NAMES):
        return jsonify({"error": "Incorrect number of features. Expected: " + str(len(FEATURE_NAMES))}), 400

    # Convert the input data into the proper format for prediction
    input_data = []
    for feature, value in zip(FEATURE_NAMES, data.values()):
        if feature == "Blood Group":
            if value in blood_group_mapping:
                input_data.append(blood_group_mapping[value])
            else:
                return jsonify({"error": f"Invalid Blood Group value: {value}"}), 400
        elif feature == "Cycle(R/I)":
            if value.upper() == 'R':
                input_data.append(2)
            elif value.upper() == 'I':
                input_data.append(5)
            else:
                return jsonify({"error": "Invalid value for 'Cycle(R/I)'. Must be 'R' or 'I'"}), 400
        elif feature in ["Pregnant(Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
                         "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)"]:
            if value.upper() in yn_mapping:
                input_data.append(yn_mapping[value.upper()])
            else:
                return jsonify({"error": f"Invalid value for '{feature}'. Must be 'Y' or 'N'"}), 400
        else:
            try:
                # Attempt to convert numeric features
                input_data.append(float(value))
            except ValueError:
                return jsonify({"error": f"Invalid value for '{feature}'. Expected a numeric value."}), 400

    # Convert input_data to numpy array and reshape for prediction
    input_array = np.array(input_data).reshape(1, -1)

    # Handle missing values using the imputer
    input_array = imputer.transform(input_array)

    # Scale the input data
    input_scaled = scaler.transform(input_array)

    # Predict using the trained model
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Return the prediction and probabilities
    pc_pos_probability = prediction_proba[0][1]  # Probability of PCOS
    pc_neg_probability = prediction_proba[0][0]  # Probability of No PCOS

    result = {
        "prediction": "PCOS" if prediction[0] == 1 else "No PCOS",
        "PCOS_probability": pc_pos_probability * 100,
        "No_PCOS_probability": pc_neg_probability * 100
    }

    return jsonify(result), 200


# Run the Flask app
if __name__ == '__main__':
    # Make sure to set debug to False in production
    app.run(debug=True)
