from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pubchempy as pcp  # New library for name lookup
import os
import sys

# --- ROBUST PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'dexa_interaction_model.pkl')
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')

# --- FLASK APP ---
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# --- GLOBALS ---
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
MODEL = None

def load_model():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        MODEL = joblib.load(MODEL_PATH)
        return True
    except:
        return False

def resolve_input_to_smiles(user_input):
    """
    Smart Resolver:
    1. Checks if input is already a valid SMILES string.
    2. If not, tries to fetch it from PubChem as a drug name.
    """
    user_input = user_input.strip()
    
    # Strategy 1: Check if it is already a SMILES code
    if Chem.MolFromSmiles(user_input) is not None:
        return user_input, "SMILES (Direct Input)"

    # Strategy 2: Search PubChem by Name
    try:
        print(f"🔍 Searching PubChem for: {user_input}")
        compounds = pcp.get_compounds(user_input, namespace='name')
        if compounds:
            # Return the isomeric SMILES (most accurate structure)
            return compounds[0].isomeric_smiles, f"Name Resolved: {user_input}"
    except Exception as e:
        print(f"PubChem Error: {e}")
        
    return None, None

def preprocess_structure(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        fp = MORGAN_GEN.GetFingerprintAsNumPy(mol)
        return fp.reshape(1, -1)
    except:
        return None

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    data = request.get_json() or request.form
    user_input = data.get('smiles', '').strip()

    if not user_input:
        return jsonify({'error': 'Please enter a Drug Name or SMILES'}), 400

    # 1. Resolve Name -> SMILES
    smiles, source_note = resolve_input_to_smiles(user_input)
    
    if smiles is None:
        return jsonify({'error': f"Could not find structure for '{user_input}'. Check spelling or try a SMILES string."}), 404

    # 2. Preprocess
    features = preprocess_structure(smiles)
    if features is None:
        return jsonify({'error': 'Structure found but could not be processed.'}), 400

    # 3. Predict
    try:
        prediction = MODEL.predict(features)[0]
        probabilities = MODEL.predict_proba(features)[0]
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        result = {
            'success': True,
            'original_input': user_input,
            'resolved_smiles': smiles,
            'source_note': source_note,
            'risk_level': 'HIGH' if prediction == 1 else 'LOW',
            'confidence_score': float(f"{confidence * 100:.2f}")
        }

        if prediction == 1:
            result['message'] = "Interaction Detected"
            result['hint'] = "This drug likely competes with Dexamethasone for the Glucocorticoid Receptor."
            result['css_class'] = "danger"
        else:
            result['message'] = "Safe / No Interaction"
            result['hint'] = "This drug does not appear to bind to the target receptor."
            result['css_class'] = "success"

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if load_model():
        print("🚀 Server Ready! Open http://127.0.0.1:5000")
        app.run(debug=True, port=5000)
    else:
        print("❌ Failed to load model. Run 'python scr/train.py' first.")