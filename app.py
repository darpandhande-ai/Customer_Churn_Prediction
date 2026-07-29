import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the AdaBoost Model
MODEL_PATH = 'Adaboost_model.pkl'
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("AdaBoost Model loaded successfully.")
    else:
        print(f"Warning: '{MODEL_PATH}' not found in current directory.")
except Exception as e:
    print(f"Error loading model: {e}")

# Exact features reconstructed from model metadata
FEATURE_CONFIG = {
    "Age": {"type": "number", "min": 18, "max": 100, "default": 35, "step": 1, "unit": "years"},
    "Gender": {"type": "select", "options": [0, 1], "labels": ["Female", "Male"], "default": 0},
    "Tenure": {"type": "number", "min": 0, "max": 120, "default": 24, "step": 1, "unit": "months"},
    "Usage Frequency": {"type": "number", "min": 0, "max": 30, "default": 15, "step": 1, "unit": "days/mo"},
    "Support Calls": {"type": "number", "min": 0, "max": 20, "default": 2, "step": 1, "unit": "calls"},
    "Payment Delay": {"type": "number", "min": 0, "max": 60, "default": 5, "step": 1, "unit": "days"},
    "Subscription Type": {"type": "select", "options": [0, 1, 2], "labels": ["Basic", "Standard", "Premium"], "default": 1},
    "Contract Length": {"type": "select", "options": [0, 1, 2], "labels": ["Monthly", "Quarterly", "Annual"], "default": 1},
    "Total Spend": {"type": "number", "min": 0, "max": 10000, "default": 1200, "step": 10, "unit": "$"},
    "Last Interaction": {"type": "number", "min": 0, "max": 30, "default": 7, "step": 1, "unit": "days ago"}
}

@app.route('/')
def index():
    # Pass zip to template context for options/labels loop
    return render_template('index.html', feature_config=FEATURE_CONFIG, zip=zip)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server. Ensure Adaboost_model.pkl exists.'}), 500

    try:
        data = request.json or {}
        
        # Extract features in the exact expected order
        features = [
            float(data.get('Age', 35)),
            float(data.get('Gender', 0)),
            float(data.get('Tenure', 24)),
            float(data.get('Usage Frequency', 15)),
            float(data.get('Support Calls', 2)),
            float(data.get('Payment Delay', 5)),
            float(data.get('Subscription Type', 1)),
            float(data.get('Contract Length', 1)),
            float(data.get('Total Spend', 1200)),
            float(data.get('Last Interaction', 7))
        ]

        input_array = np.array([features])
        
        # Execute prediction
        prediction = int(model.predict(input_array)[0])
        probabilities = model.predict_proba(input_array)[0].tolist()

        # Extract feature importances
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_.tolist()
        else:
            feature_importances = [0.1] * len(FEATURE_CONFIG)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probabilities': probabilities,
            'feature_names': list(FEATURE_CONFIG.keys()),
            'feature_importances': feature_importances,
            'estimator_count': len(getattr(model, 'estimators_', [])) or 50
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
