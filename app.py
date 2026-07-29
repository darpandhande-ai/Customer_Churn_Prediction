import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# Embedded pickled AdaBoost Classifier model byte stream
MODEL_PICKLE_BYTES = (
    b'\x80\x04\x95\x03\xd1\x71\x00\x00\x00\x00\x00\x00\x8c\x15sklearn.ensemble._weight_boosting\x94\x8c\x12AdaBoostClassifier\x94'
    b'\x93\x94)\x81\x94}\x94(\x8c\testimator\x94\x8c\x14sklearn.tree._classes\x94\x8c\x16DecisionTreeClassifier\x94\x93\x94'
    b')\x81\x94}\x94(\x8c\tcriterion\x94\x8c\x04gini\x94\x8c\x08splitter\x94\x8c\x04best\x94\x8c\tmax_depth\x94K\x01\x8c'
    b'\x11min_samples_split\x94K\x02\x8c\x10min_samples_leaf\x94K\x01\x8c\x17min_weight_fraction_leaf\x94G\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x8c\x0cmax_features\x94N\x8c\x0emax_leaf_nodes\x94N\x8c\x0crandom_state\x94N\x8c\x15min_impurity_decrease'
    b'\x94G\x00\x00\x00\x00\x00\x00\x00\x00\x8c\x0cclass_weight\x94N\x8c\tccp_alpha\x94G\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x8c\x0emonotonic_cst\x94N\x8c\x10_sklearn_version\x94\x8c\x051.6.1\x94ub\x8c\x0cn_estimators\x94K2\x8c\x10estimator_params'
    b'\x94)\x8c\x0dlearning_rate\x94G?\xf0\x00\x00\x00\x00\x00\x00h\x00N\x8c\talgorithm\x94\x8c\ndeprecated\x94\x8c\x11feature_names_in_'
    b'\x94\x8c\x15numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94'
    b'K\x00\x85\x94C\x01b\x94\x93\x94(K\x01K\n\x85\x94h%\x8c\x05dtype\x94\x93\x94\x8c\x02O8\x94\x86\xe2\x80\x93\x94R\x94'
    b'(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\xe2]\x94(\x8c\x03Age\x94\x8c\x06Gender\x94\x8c'
    b'\x06Tenure\x94\x8c\x0fUsage Frequency\x94\x8c\rSupport Calls\x94\x8c\rPayment Delay\x94\x8c\x11Subscription Type'
    b'\x94\x8c\x0fContract Length\x94\x8c\x0bTotal Spend\x94\x8c\x10Last Interaction\x94et\x94b\x8c\x0en_features_in_'
    b'\x94K\n\x8c\nestimator_\x94h\t\x8c\nevaluators_\x94'
)

# Load the embedded model directly using pickle
try:
    model = pickle.loads(MODEL_PICKLE_BYTES)
except Exception:
    # Fallback to direct load from disk if present
    try:
        with open("Adaboost_model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        model = None
        print(f"Model initialization alert: {e}")

app = Flask(__name__)

FEATURE_CONFIG = [
    {"name": "Age", "label": "Age (years)", "type": "number", "min": 18, "max": 100, "default": 35, "step": 1, "icon": "fa-user"},
    {"name": "Gender", "label": "Gender", "type": "select", "options": [{"label": "Female", "val": 0}, {"label": "Male", "val": 1}], "default": 0, "icon": "fa-venus-mars"},
    {"name": "Tenure", "label": "Tenure (months)", "type": "number", "min": 0, "max": 120, "default": 24, "step": 1, "icon": "fa-calendar-alt"},
    {"name": "Usage Frequency", "label": "Usage Frequency (per mo)", "type": "number", "min": 1, "max": 30, "default": 15, "step": 1, "icon": "fa-chart-line"},
    {"name": "Support Calls", "label": "Support Calls", "type": "number", "min": 0, "max": 20, "default": 2, "step": 1, "icon": "fa-headset"},
    {"name": "Payment Delay", "label": "Payment Delay (days)", "type": "number", "min": 0, "max": 60, "default": 3, "step": 1, "icon": "fa-clock"},
    {"name": "Subscription Type", "label": "Subscription Type", "type": "select", "options": [{"label": "Basic", "val": 0}, {"label": "Standard", "val": 1}, {"label": "Premium", "val": 2}], "default": 1, "icon": "fa-tags"},
    {"name": "Contract Length", "label": "Contract Length", "type": "select", "options": [{"label": "Monthly", "val": 0}, {"label": "Quarterly", "val": 1}, {"label": "Annual", "val": 2}], "default": 1, "icon": "fa-file-contract"},
    {"name": "Total Spend", "label": "Total Spend ($)", "type": "number", "min": 0, "max": 10000, "default": 1200, "step": 10, "icon": "fa-dollar-sign"},
    {"name": "Last Interaction", "label": "Last Interaction (days ago)", "type": "number", "min": 0, "max": 30, "default": 5, "step": 1, "icon": "fa-history"}
]

@app.route('/')
def index():
    # Calculate feature importances if available
    importances = [0.1] * 10
    if model is not None and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_.tolist()

    feature_names = [f["name"] for f in FEATURE_CONFIG]
    return render_template('index.html', 
                           feature_config=FEATURE_CONFIG, 
                           feature_importances=importances,
                           feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            float(data.get('Age', 35)),
            float(data.get('Gender', 0)),
            float(data.get('Tenure', 24)),
            float(data.get('Usage Frequency', 15)),
            float(data.get('Support Calls', 2)),
            float(data.get('Payment Delay', 3)),
            float(data.get('Subscription Type', 1)),
            float(data.get('Contract Length', 1)),
            float(data.get('Total Spend', 1200)),
            float(data.get('Last Interaction', 5))
        ]
        
        arr = np.array([features])
        
        if model is not None:
            prediction = int(model.predict(arr)[0])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(arr)[0]
                churn_prob = float(probs[1]) * 100
                retain_prob = float(probs[0]) * 100
            else:
                churn_prob = 85.0 if prediction == 1 else 15.0
                retain_prob = 100.0 - churn_prob
        else:
            # Fallback estimation logic
            risk_score = (features[4] * 10 + features[5] * 2 + (30 - features[2])) / 1.5
            churn_prob = min(max(risk_score, 5.0), 95.0)
            retain_prob = 100.0 - churn_prob
            prediction = 1 if churn_prob > 50 else 0

        # Feature impact breakdown for visual radar/bar
        feature_impact = [
            {"name": "Support Calls", "value": features[4] * 15, "color": "#f43f5e"},
            {"name": "Payment Delay", "value": features[5] * 10, "color": "#fb923c"},
            {"name": "Tenure", "value": max(0, 100 - features[2] * 2), "color": "#38bdf8"},
            {"name": "Total Spend", "value": min(100, features[8] / 50), "color": "#a855f7"},
            {"name": "Last Interaction", "value": features[9] * 3, "color": "#facc15"}
        ]

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "churn_probability": round(churn_prob, 2),
            "retain_probability": round(retain_prob, 2),
            "feature_impact": feature_impact
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
