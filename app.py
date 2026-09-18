import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Model path definition
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Adaboost_model.pkl")

# Load model safely
model = None
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found.")
except Exception as e:
    print(f"Error loading model: {e}")

# HTML/CSS/JS single-file responsive template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Customer Churn Prediction & Analytics</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
    
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            /* Colorful Theme Variables - Dark Default */
            --bg-base: #0B0F19;
            --bg-surface: #111827;
            --bg-card: rgba(31, 41, 55, 0.65);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-primary: #F9FAFB;
            --text-secondary: #9CA3AF;
            
            --accent-primary: #8B5CF6;
            --accent-secondary: #EC4899;
            --accent-tertiary: #3B82F6;
            --accent-cyan: #06B6D4;
            --accent-emerald: #10B981;
            --accent-amber: #F59E0B;
            --accent-rose: #F43F5E;
            
            --gradient-main: linear-gradient(135deg, #8B5CF6 0%, #EC4899 50%, #3B82F6 100%);
            --gradient-glow: linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(236, 72, 153, 0.25));
            --shadow-glass: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            --radius-lg: 18px;
            --radius-md: 12px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        [data-theme="light"] {
            --bg-base: #F3F4F6;
            --bg-surface: #FFFFFF;
            --bg-card: rgba(255, 255, 255, 0.85);
            --border-color: rgba(0, 0, 0, 0.08);
            --text-primary: #111827;
            --text-secondary: #4B5563;
            --shadow-glass: 0 10px 30px -10px rgba(0, 0, 0, 0.1);
        }

        /* Preset Dynamic Palettes */
        [data-palette="cyberpunk"] {
            --accent-primary: #FF007F;
            --accent-secondary: #00F0FF;
            --gradient-main: linear-gradient(135deg, #FF007F 0%, #7928CA 50%, #00F0FF 100%);
        }

        [data-palette="emerald"] {
            --accent-primary: #059669;
            --accent-secondary: #10B981;
            --gradient-main: linear-gradient(135deg, #059669 0%, #10B981 50%, #06B6D4 100%);
        }

        [data-palette="sunset"] {
            --accent-primary: #FF512F;
            --accent-secondary: #DD2476;
            --gradient-main: linear-gradient(135deg, #FF512F 0%, #F09819 50%, #DD2476 100%);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
            transition: background-color 0.3s, border-color 0.3s, color 0.3s;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            background-image: 
                radial-gradient(at 10% 10%, rgba(139, 92, 246, 0.15) 0px, transparent 50%),
                radial-gradient(at 90% 90%, rgba(236, 72, 153, 0.12) 0px, transparent 50%),
                radial-gradient(at 50% 50%, rgba(59, 130, 246, 0.1) 0px, transparent 50%);
            background-attachment: fixed;
        }

        /* App Bar */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 2rem;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
            backdrop-filter: blur(12px);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand i {
            font-size: 1.5rem;
            color: var(--accent-primary);
            -webkit-text-fill-color: initial;
        }

        .controls-group {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .theme-selector {
            display: flex;
            gap: 6px;
            background: rgba(0,0,0,0.15);
            padding: 4px;
            border-radius: 30px;
            border: 1px solid var(--border-color);
        }

        .theme-dot {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid transparent;
            transition: var(--transition);
        }

        .theme-dot.active {
            transform: scale(1.2);
            border-color: var(--text-primary);
        }

        .dot-default { background: linear-gradient(135deg, #8B5CF6, #EC4899); }
        .dot-cyberpunk { background: linear-gradient(135deg, #FF007F, #00F0FF); }
        .dot-emerald { background: linear-gradient(135deg, #059669, #10B981); }
        .dot-sunset { background: linear-gradient(135deg, #FF512F, #DD2476); }

        .btn-icon {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            width: 38px;
            height: 38px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
        }

        .btn-icon:hover {
            transform: rotate(15deg) scale(1.05);
            border-color: var(--accent-primary);
        }

        /* Layout Grid */
        .container {
            max-width: 1440px;
            margin: 2rem auto;
            padding: 0 1.5rem;
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.75rem;
            box-shadow: var(--shadow-glass);
            position: relative;
            overflow: hidden;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: var(--gradient-main);
        }

        .card-header {
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .card-title i {
            color: var(--accent-primary);
        }

        /* Form Inputs Grid */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.2rem;
        }

        @media (max-width: 640px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        label {
            font-size: 0.825rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
        }

        input, select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: rgba(0, 0, 0, 0.15);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            transition: var(--transition);
        }

        input:focus, select:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25);
            background: rgba(0, 0, 0, 0.25);
        }

        .btn-submit {
            grid-column: span 2;
            margin-top: 1rem;
            padding: 1rem;
            background: var(--gradient-main);
            border: none;
            border-radius: var(--radius-md);
            color: white;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
            transition: var(--transition);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(236, 72, 153, 0.5);
        }

        /* Dashboard Analysis Side */
        .dashboard-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        /* Metric Highlights Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 1rem;
            text-align: center;
        }

        .metric-card .value {
            font-size: 1.5rem;
            font-weight: 800;
            margin: 4px 0;
            font-family: 'Space Grotesk', sans-serif;
        }

        .metric-card .label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        /* Result Visual Banner */
        .result-banner {
            border-radius: var(--radius-md);
            padding: 1.5rem;
            text-align: center;
            background: var(--gradient-glow);
            border: 1px solid var(--accent-primary);
            animation: pulseGlow 3s infinite alternate;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        @keyframes pulseGlow {
            0% { box-shadow: 0 0 15px rgba(139, 92, 246, 0.2); }
            100% { box-shadow: 0 0 30px rgba(236, 72, 153, 0.4); }
        }

        .result-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
        }

        .result-value {
            font-size: 2rem;
            font-weight: 800;
            font-family: 'Space Grotesk', sans-serif;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 700;
        }

        .badge-danger { background: rgba(244, 63, 94, 0.2); color: var(--accent-rose); border: 1px solid var(--accent-rose); }
        .badge-success { background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); border: 1px solid var(--accent-emerald); }

        /* Chart Boxes */
        .chart-box {
            background: rgba(0, 0, 0, 0.2);
            border-radius: var(--radius-md);
            padding: 1.25rem;
            border: 1px solid var(--border-color);
            margin-top: 1rem;
        }

        .chart-header {
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* SVG Bar Chart Styling */
        .bar-chart {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .bar-item {
            display: grid;
            grid-template-columns: 120px 1fr 50px;
            align-items: center;
            gap: 10px;
            font-size: 0.8rem;
        }

        .bar-label {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            color: var(--text-secondary);
        }

        .bar-track {
            height: 10px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 5px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 5px;
            background: var(--gradient-main);
            width: 0%;
            transition: width 1s cubic-bezier(0.1, 0.5, 0.1, 1);
        }

        .bar-value {
            text-align: right;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }

        /* Analytics KPI Cards */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }

        .kpi-card {
            background: rgba(255, 255, 255, 0.02);
            border-radius: var(--radius-md);
            padding: 1rem;
            border: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .kpi-icon {
            width: 42px;
            height: 42px;
            border-radius: 10px;
            background: var(--gradient-glow);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-primary);
            font-size: 1.2rem;
        }

        /* Spinner Animation */
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <!-- Navigation Header -->
    <nav class="navbar">
        <div class="brand">
            <i class="fa-solid fa-users-gear"></i>
            <span>AI Customer Churn Prediction</span>
        </div>
        <div class="controls-group">
            <!-- Dynamic Theme Selector -->
            <div class="theme-selector" title="Choose Color Theme">
                <div class="theme-dot dot-default active" onclick="setPalette('default')"></div>
                <div class="theme-dot dot-cyberpunk" onclick="setPalette('cyberpunk')"></div>
                <div class="theme-dot dot-emerald" onclick="setPalette('emerald')"></div>
                <div class="theme-dot dot-sunset" onclick="setPalette('sunset')"></div>
            </div>
            <!-- Dark/Light Mode Toggle -->
            <button class="btn-icon" onclick="toggleMode()" title="Toggle Dark/Light Mode">
                <i class="fa-solid fa-moon" id="modeIcon"></i>
            </button>
        </div>
    </nav>

    <!-- Main Container Layout -->
    <div class="container">
        
        <!-- Form Side Inputs -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">
                    <i class="fa-solid fa-sliders"></i> Customer Parameters
                </div>
                <span style="font-size: 0.75rem; color: var(--accent-cyan); background: rgba(6, 182, 212, 0.1); padding: 4px 8px; border-radius: 6px;">10 Inputs</span>
            </div>

            <form id="predictionForm">
                <div class="form-grid">
                    
                    <div class="form-group">
                        <label>Age</label>
                        <input type="number" id="Age" name="Age" value="38" min="18" max="100" required>
                    </div>

                    <div class="form-group">
                        <label>Gender</label>
                        <select id="Gender" name="Gender">
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Tenure (Months)</label>
                        <input type="number" id="Tenure" name="Tenure" value="18" min="0" max="120" required>
                    </div>

                    <div class="form-group">
                        <label>Usage Frequency</label>
                        <input type="number" id="Usage Frequency" name="Usage Frequency" value="8" min="1" max="100" required>
                    </div>

                    <div class="form-group">
                        <label>Support Calls</label>
                        <input type="number" id="Support Calls" name="Support Calls" value="4" min="0" max="20" required>
                    </div>

                    <div class="form-group">
                        <label>Payment Delay (Days)</label>
                        <input type="number" id="Payment Delay" name="Payment Delay" value="12" min="0" max="60" required>
                    </div>

                    <div class="form-group">
                        <label>Subscription Type</label>
                        <select id="Subscription Type" name="Subscription Type">
                            <option value="Basic">Basic</option>
                            <option value="Standard">Standard</option>
                            <option value="Premium">Premium</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Contract Length</label>
                        <select id="Contract Length" name="Contract Length">
                            <option value="Monthly">Monthly</option>
                            <option value="Quarterly">Quarterly</option>
                            <option value="Annual">Annual</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Total Spend ($)</label>
                        <input type="number" step="0.01" id="Total Spend" name="Total Spend" value="320.00" min="0" required>
                    </div>

                    <div class="form-group">
                        <label>Last Interaction (Days)</label>
                        <input type="number" id="Last Interaction" name="Last Interaction" value="22" min="0" max="365" required>
                    </div>

                    <button type="submit" class="btn-submit">
                        <span id="btnText">Analyze Churn Probability</span>
                        <div class="spinner" id="btnSpinner"></div>
                        <i class="fa-solid fa-chart-line"></i>
                    </button>

                </div>
            </form>
        </div>

        <!-- Analytics Dashboard Side -->
        <div class="dashboard-container">
            
            <!-- Result Overview Banner -->
            <div class="card result-banner">
                <div class="result-title">AI Customer Churn Status</div>
                <div class="result-value" id="resultLabel">Ready for Analysis</div>
                <div id="statusBadge" class="status-badge badge-success" style="display:none;">
                    <i class="fa-solid fa-shield-halved"></i> <span id="badgeText">Low Risk</span>
                </div>
            </div>

            <!-- Analytics Dashboard Box -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fa-solid fa-chart-pie"></i> Analytics Dashboard
                    </div>
                    <span style="font-size: 0.75rem; color: var(--accent-emerald); background: rgba(16, 185, 129, 0.1); padding: 4px 8px; border-radius: 6px;">Live Metrics</span>
                </div>

                <!-- Top Metric Cards -->
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">Confidence</div>
                        <div class="value" style="color: var(--accent-emerald);" id="metricConfidence">--%</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Churn Risk Index</div>
                        <div class="value" style="color: var(--accent-rose);" id="metricRisk">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Customer Value</div>
                        <div class="value" style="color: var(--accent-cyan);" id="metricLTV">$0.00</div>
                    </div>
                </div>

                <!-- KPI Analytics Summary Grid -->
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-icon"><i class="fa-solid fa-headset"></i></div>
                        <div>
                            <div style="font-size: 0.75rem; color: var(--text-secondary);">Support Burden</div>
                            <div style="font-size: 1.1rem; font-weight:700;" id="kpiSupport">Normal</div>
                        </div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-icon"><i class="fa-solid fa-credit-card"></i></div>
                        <div>
                            <div style="font-size: 0.75rem; color: var(--text-secondary);">Payment Reliability</div>
                            <div style="font-size: 1.1rem; font-weight:700;" id="kpiPayment">Good</div>
                        </div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-icon"><i class="fa-solid fa-user-clock"></i></div>
                        <div>
                            <div style="font-size: 0.75rem; color: var(--text-secondary);">Engagement Status</div>
                            <div style="font-size: 1.1rem; font-weight:700;" id="kpiEngagement">Active</div>
                        </div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-icon"><i class="fa-solid fa-file-contract"></i></div>
                        <div>
                            <div style="font-size: 0.75rem; color: var(--text-secondary);">Retention Health</div>
                            <div style="font-size: 1.1rem; font-weight:700;" id="kpiHealth">Stable</div>
                        </div>
                    </div>
                </div>

                <!-- Feature Impact Bar Chart Visualizer -->
                <div class="chart-box">
                    <div class="chart-header">
                        <span>Feature Relative Distribution</span>
                        <i class="fa-solid fa-chart-column" style="color: var(--accent-secondary);"></i>
                    </div>
                    <div class="bar-chart" id="chartContainer">
                        <div style="text-align: center; color: var(--text-secondary); padding: 1rem 0;">
                            Submit parameters to generate analytics breakdown.
                        </div>
                    </div>
                </div>

            </div>

        </div>

    </div>

    <!-- Frontend Dynamic Script -->
    <script>
        // Palette Switching Logic
        function setPalette(palette) {
            document.documentElement.setAttribute('data-palette', palette);
            document.querySelectorAll('.theme-dot').forEach(dot => dot.classList.remove('active'));
            event.target.classList.add('active');
        }

        // Dark/Light Theme Switch
        function toggleMode() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const icon = document.getElementById('modeIcon');
            
            if (currentTheme === 'dark') {
                html.setAttribute('data-theme', 'light');
                icon.className = 'fa-solid fa-sun';
            } else {
                html.setAttribute('data-theme', 'dark');
                icon.className = 'fa-solid fa-moon';
            }
        }

        // Form Submit Handler
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const btnText = document.getElementById('btnText');
            const btnSpinner = document.getElementById('btnSpinner');
            btnText.style.display = 'none';
            btnSpinner.style.display = 'block';

            const formData = new FormData(this);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if(data.status === 'success') {
                    renderDashboard(data);
                } else {
                    alert('Error: ' + data.message);
                }
            } catch (err) {
                console.error(err);
                alert('Prediction request failed!');
            } finally {
                btnText.style.display = 'inline';
                btnSpinner.style.display = 'none';
            }
        });

        // Function to Render Dashboard Analytics Dynamically
        function renderDashboard(data) {
            // Update Main Banner
            const labelEl = document.getElementById('resultLabel');
            const badgeEl = document.getElementById('statusBadge');
            const badgeText = document.getElementById('badgeText');
            
            labelEl.textContent = data.prediction_text;
            badgeEl.style.display = 'inline-flex';

            if(data.prediction === 1) {
                badgeEl.className = 'status-badge badge-danger';
                badgeText.textContent = 'High Attrition Risk';
            } else {
                badgeEl.className = 'status-badge badge-success';
                badgeText.textContent = 'Retained Customer';
            }

            // Update Top Metrics
            document.getElementById('metricConfidence').textContent = data.confidence + '%';
            document.getElementById('metricRisk').textContent = data.risk_score + '/100';
            
            const spend = parseFloat(document.getElementById('Total Spend').value);
            document.getElementById('metricLTV').textContent = '$' + spend.toFixed(2);

            // Dynamic KPI Logic Calculation
            const supportCalls = parseFloat(document.getElementById('Support Calls').value);
            const paymentDelay = parseFloat(document.getElementById('Payment Delay').value);
            const lastInteraction = parseFloat(document.getElementById('Last Interaction').value);

            document.getElementById('kpiSupport').textContent = supportCalls > 3 ? 'High Risk' : 'Normal';
            document.getElementById('kpiPayment').textContent = paymentDelay > 10 ? 'Delayed' : 'Reliable';
            document.getElementById('kpiEngagement').textContent = lastInteraction > 20 ? 'Dormant' : 'Active';
            document.getElementById('kpiHealth').textContent = data.risk_score > 50 ? 'Critical' : 'Stable';

            // Render Dynamic Bar Chart
            const chartBox = document.getElementById('chartContainer');
            chartBox.innerHTML = '';

            const features = data.features_impact;
            const maxVal = Math.max(...Object.values(features));

            for (const [key, val] of Object.entries(features)) {
                const pct = maxVal > 0 ? ((val / maxVal) * 100).toFixed(0) : 0;
                const barItem = document.createElement('div');
                barItem.className = 'bar-item';
                barItem.innerHTML = `
                    <div class="bar-label" title="${key}">${key}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: 0%"></div>
                    </div>
                    <div class="bar-value">${val}</div>
                `;
                chartBox.appendChild(barItem);

                // Bar fill animation
                setTimeout(() => {
                    barItem.querySelector('.bar-fill').style.width = pct + '%';
                }, 100);
            }
        }
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
        return jsonify({
            'status': 'error',
            'message': 'Model file (Adaboost_model.pkl) is missing or failed to load.'
        })

    try:
        # Extract inputs from form POST request
        age = float(request.form.get('Age', 0))
        gender_str = request.form.get('Gender', 'Male')
        tenure = float(request.form.get('Tenure', 0))
        usage_freq = float(request.form.get('Usage Frequency', 0))
        support_calls = float(request.form.get('Support Calls', 0))
        payment_delay = float(request.form.get('Payment Delay', 0))
        sub_type_str = request.form.get('Subscription Type', 'Basic')
        contract_str = request.form.get('Contract Length', 'Annual')
        total_spend = float(request.form.get('Total Spend', 0))
        last_interaction = float(request.form.get('Last Interaction', 0))

        # Perform Label Encoding/Mapping for Categorical Variables
        gender = 1 if gender_str == 'Male' else 0
        
        sub_map = {'Basic': 0, 'Standard': 1, 'Premium': 2}
        subscription_type = sub_map.get(sub_type_str, 0)

        contract_map = {'Monthly': 0, 'Quarterly': 1, 'Annual': 2}
        contract_length = contract_map.get(contract_str, 0)

        # Build numpy array matching model order:
        # ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction']
        features = np.array([[
            age, gender, tenure, usage_freq, support_calls,
            payment_delay, subscription_type, contract_length,
            total_spend, last_interaction
        ]])

        # Execute Prediction
        prediction = model.predict(features)[0]
        
        # Calculate Probabilities / Risk Score
        confidence = 88.5
        risk_score = 25
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            confidence = round(float(np.max(probs)) * 100, 1)
            risk_score = round(float(probs[1]) * 100, 1) if len(probs) > 1 else int(prediction * 100)

        pred_text = "Customer Churn Likely" if prediction == 1 else "Customer Retained"

        # Feature impact map for charts
        features_dict = {
            'Age': age,
            'Tenure': tenure,
            'Usage Freq': usage_freq,
            'Support Calls': support_calls,
            'Payment Delay': payment_delay,
            'Total Spend': total_spend,
            'Last Interact': last_interaction
        }

        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'prediction_text': pred_text,
            'confidence': confidence,
            'risk_score': risk_score,
            'features_impact': features_dict
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
