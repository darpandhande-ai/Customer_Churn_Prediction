import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sfsboodt_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# HTML & Inline CSS Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Analytics Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(255, 255, 255, 0.15);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header h1 {
            font-size: 2.25rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        input, select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 10px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        input:focus, select:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }

        select option {
            background-color: #0f172a;
            color: #f8fafc;
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1rem;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }

        .btn-submit:hover {
            background: var(--accent-hover);
            transform: translateY(-1px);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
            display: none;
            animation: fadeIn 0.3s ease;
        }

        .result-box.success {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
        }

        .result-box.error {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #f87171;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Customer Prediction Portal</h1>
            <p>Enter the customer metrics below to generate a model prediction</p>
        </div>

        <form id="predictionForm" class="form-grid">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="Age" placeholder="e.g. 35" required min="0">
            </div>

            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="Gender" required>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>

            <div class="form-group">
                <label for="tenure">Tenure (Months)</label>
                <input type="number" id="tenure" name="Tenure" placeholder="e.g. 12" required min="0">
            </div>

            <div class="form-group">
                <label for="usage">Usage Frequency</label>
                <input type="number" id="usage" name="Usage Frequency" placeholder="e.g. 15" required min="0">
            </div>

            <div class="form-group">
                <label for="calls">Support Calls</label>
                <input type="number" id="calls" name="Support Calls" placeholder="e.g. 2" required min="0">
            </div>

            <div class="form-group">
                <label for="delay">Payment Delay (Days)</label>
                <input type="number" id="delay" name="Payment Delay" placeholder="e.g. 0" required min="0">
            </div>

            <div class="form-group">
                <label for="subscription">Subscription Type</label>
                <select id="subscription" name="Subscription Type" required>
                    <option value="0">Basic</option>
                    <option value="1">Standard</option>
                    <option value="2">Premium</option>
                </select>
            </div>

            <div class="form-group">
                <label for="contract">Contract Length</label>
                <select id="contract" name="Contract Length" required>
                    <option value="0">Monthly</option>
                    <option value="1">Quarterly</option>
                    <option value="2">Annual</option>
                </select>
            </div>

            <div class="form-group">
                <label for="spend">Total Spend ($)</label>
                <input type="number" step="0.01" id="spend" name="Total Spend" placeholder="e.g. 450.00" required min="0">
            </div>

            <div class="form-group">
                <label for="interaction">Last Interaction (Days)</label>
                <input type="number" id="interaction" name="Last Interaction" placeholder="e.g. 5" required min="0">
            </div>

            <button type="submit" class="btn-submit">Run Prediction</button>
        </form>

        <div id="result" class="result-box"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const resultBox = document.getElementById('result');
            resultBox.style.display = 'none';

            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const res = await response.json();
                
                if (response.ok) {
                    resultBox.className = 'result-box success';
                    resultBox.innerText = `Prediction Output: ${res.prediction}`;
                } else {
                    resultBox.className = 'result-box error';
                    resultBox.innerText = res.error || 'An error occurred during prediction.';
                }
            } catch (err) {
                resultBox.className = 'result-box error';
                resultBox.innerText = 'Failed to connect to backend server.';
            }

            resultBox.style.display = 'block';
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model pickle file not loaded properly'}), 500

    try:
        data = request.get_json()
        
        # Expected feature order
        feature_order = [
            'Age', 'Gender', 'Tenure', 'Usage Frequency', 
            'Support Calls', 'Payment Delay', 'Subscription Type', 
            'Contract Length', 'Total Spend', 'Last Interaction'
        ]
        
        features = [float(data[feature]) for feature in feature_order]
        prediction = model.predict([features])[0]

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
