import os
import sys
import requests # Need this to call Google
from flask import Flask, render_template, request, jsonify

# --- LOAD SECRETS ---
# This tries to load .env file locally, or system vars on Render
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'scr'))

try:
    from predict_phase2 import predict_drug, analyze_interaction
except ImportError:
    print("❌ ERROR: Check 'scr/predict_phase2.py'.")
    sys.exit(1)

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('dashboard_phase2.html')

# --- EXISTING PREDICTION ROUTES ---
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    return jsonify(predict_drug(data.get('smiles')))

@app.route('/predict_interaction', methods=['POST'])
def interaction_endpoint():
    data = request.get_json()
    return jsonify(analyze_interaction(data.get('drug_a'), data.get('drug_b')))

# --- NEW SECURE AI ROUTE ---
@app.route('/ask_ai', methods=['POST'])
def ask_ai_endpoint():
    """
    Acts as a secure proxy. The frontend calls this, 
    and this calls Google using the hidden server-side key.
    """
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Server Configuration Error: API Key missing'}), 500
        
    data = request.get_json()
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    # Call Google Gemini from the Server (Secure)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"
    payload = { "contents": [{ "parts": [{ "text": prompt }] }] }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': f"AI Service Error: {str(e)}"}), 500

if __name__ == '__main__':
    print("--- 🚀 DexaGen-AI Secure Server Running ---")
    app.run(debug=True)