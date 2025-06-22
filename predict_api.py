from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)
model = joblib.load('depression_logreg_model.pkl')
scaler = joblib.load('depression_scaler.pkl')

FEATURES = [
    "Gender",
    "Age",
    "Academic Pressure",
    "Study Satisfaction",
    "Sleep Duration",
    "Dietary Habits",
    "Have you ever had suicidal thoughts ?",
    "Study Hours",
    "Financial Stress",
    "Family History of Mental Illness"
]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = [data[feature] for feature in FEATURES]
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400

    features_np = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_np)
    prediction = model.predict(features_scaled)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)