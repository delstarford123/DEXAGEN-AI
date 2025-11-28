import os
import sys
from flask import Flask, render_template, request, jsonify

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'scr'))

# --- IMPORT LOGIC ---
try:
    from predict_phase2 import predict_drug, analyze_interaction
except ImportError:
    print("❌ ERROR: Check 'scr/predict_phase2.py'.")
    sys.exit(1)

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    try:
        return render_template('dashboard_phase2.html')
    except Exception as e:
        return f"<h2>Template Error</h2><p>{e}</p>"

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    return jsonify(predict_drug(data.get('smiles')))

@app.route('/predict_interaction', methods=['POST'])
def interaction_endpoint():
    data = request.get_json()
    return jsonify(analyze_interaction(data.get('drug_a'), data.get('drug_b')))

if __name__ == '__main__':
    print("--- 🚀 DexaGen-AI Research Assistant Running ---")
    print("    Access: http://127.0.0.1:5000")
    app.run(debug=True)