import os
import sys
import joblib
import numpy as np
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem
import requests

# Try importing PubChem for name resolution
try:
    import pubchempy as pcp
    HAS_PUBCHEM = True
except ImportError:
    HAS_PUBCHEM = False

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'stress_receptor_model.pkl')

# --- EXPERT KNOWLEDGE BASE ---
INTERACTION_DB = {
    frozenset(['Dexamethasone', 'Aspirin']): {
        "title": "SYNERGISTIC MUCOSAL EROSION PATHWAY",
        "risk": "HIGH (Gastric Hemorrhage)",
        "mechanism": (
            "PHARMACODYNAMIC CASCADE ANALYSIS:\n"
            "This combination initiates a dual-mechanism blockade of the Prostaglandin synthesis pathway:\n\n"
            "1. ENZYMATIC INHIBITION (Cytosol/ER): Aspirin irreversibly acetylates the Serine-530 residue of the COX-1 enzyme. This immediately halts the conversion of Arachidonic Acid into cytoprotective Prostaglandins (PGE2).\n\n"
            "2. GENOMIC TRANSREPRESSION (Nucleus): Dexamethasone-bound NR3C1 receptors translocate to the nucleus and physically interact with NF-κB. This inhibits the transcription of COX-2 mRNA.\n\n"
            "3. PHYSIOLOGICAL OUTCOME: The simultaneous lack of COX-1 (Housekeeping) and COX-2 (Inducible) activity strips the gastric mucosa of its protective bicarbonate layer."
        )
    },
    frozenset(['Dexamethasone', 'Cortisol']): {
        "title": "COMPETITIVE RECEPTOR DISPLACEMENT",
        "risk": "MODERATE (Adrenal Insufficiency)",
        "mechanism": (
            "PHARMACOKINETIC COMPETITION ANALYSIS:\n"
            "Comparison of Exogenous vs. Endogenous Ligand Binding:\n\n"
            "1. AFFINITY DIFFERENTIAL: Dexamethasone exhibits a dissociation constant (Kd) approximately 30-fold lower than endogenous Cortisol.\n\n"
            "2. COMPETITIVE EXCLUSION: Due to superior affinity and plasma half-life, Dexamethasone saturates the cytosolic glucocorticoid receptors, effectively displacing native Cortisol.\n\n"
            "3. NEGATIVE FEEDBACK: The potent Dex-NR3C1 complex hyper-stimulates GREs in the Pituitary gland, suppressing ACTH transcription and halting natural adrenal function."
        )
    }
}

def load_model():
    if not os.path.exists(MODEL_PATH): return None
    return joblib.load(MODEL_PATH)

def get_pubchem_3d(identifier):
    """Fetches professional 3D coordinates from PubChem API."""
    try:
        # Try Name First
        cids = pcp.get_cids(identifier, namespace='name')
        # Try SMILES if name fails
        if not cids: cids = pcp.get_cids(identifier, namespace='smiles')
        
        if cids:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids[0]}/record/SDF/?record_type=3d&response_type=display"
            response = requests.get(url, timeout=5)
            if response.status_code == 200: return response.text
    except:
        pass
    return None

def resolve_name(user_input):
    """
    Smart Resolver: Ensures we always return a Readable Name.
    """
    user_input = str(user_input).strip()
    
    # CASE 1: Input is a SMILES code
    if Chem.MolFromSmiles(user_input):
        smiles = user_input
        display_name = "Custom Molecule" # Default
        
        # Try Reverse Lookup (SMILES -> Name)
        if HAS_PUBCHEM:
            try:
                compounds = pcp.get_compounds(user_input, namespace='smiles')
                if compounds and compounds[0].synonyms:
                    display_name = compounds[0].synonyms[0] # e.g. "Aspirin"
            except:
                pass
        return smiles, display_name

    # CASE 2: Input is a Name
    if HAS_PUBCHEM:
        try:
            compounds = pcp.get_compounds(user_input, namespace='name')
            if compounds:
                # Use User's Input Title-Cased (Cleanest for UI)
                return compounds[0].isomeric_smiles, user_input.title()
        except:
            pass
            
    return None, user_input

def featurize_smiles(smiles):
    try:
        featurizer = dc.feat.CircularFingerprint(size=1024, radius=2)
        features = featurizer.featurize([smiles])
        if len(features) > 0 and features[0].shape == (1024,): return features[0].reshape(1, -1)
    except: pass
    return None

def generate_mechanism_report(is_active, confidence, name):
    """Generates the Educational Biological Report."""
    if is_active:
        return (
            f"RESEARCH ANALYSIS FOR {name.upper()}:\n\n"
            f"1. MEMBRANE DIFFUSION: This lipophilic compound ({confidence:.1%} binding probability) passively traverses the phospholipid bilayer.\n"
            f"2. CYTOSOLIC BINDING: It targets the NR3C1 (Glucocorticoid Receptor) in the cytoplasm, triggering the dissociation of Heat Shock Proteins (HSP90).\n"
            f"3. NUCLEAR TRANSLOCATION: The Ligand-Receptor complex dimerizes and actively translocates through the nuclear pore.\n"
            f"4. GENOMIC TRANSCRIPTION: The complex binds to Glucocorticoid Response Elements (GREs) on the DNA, initiating the transcription of anti-inflammatory proteins (Annexin A1)."
        )
    else:
        return (
            f"RESEARCH ANALYSIS FOR {name.upper()}:\n\n"
            f"Structural analysis indicates a lack of pharmacophore features required for the NR3C1 Ligand Binding Domain (LBD). "
            f"The molecule likely remains extracellular or interacts with alternative cytosolic pathways without triggering the specific nuclear translocation sequence."
        )

def predict_drug(user_input):
    """Single Drug Analysis"""
    model = load_model()
    if not model: return {"error": "Model missing."}

    mol_3d = get_pubchem_3d(user_input)
    smiles, name = resolve_name(user_input)
    
    if not smiles: return {"error": f"Could not identify '{user_input}'."}

    features = featurize_smiles(smiles)
    if features is None: return {"error": "Invalid structure."}

    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    confidence = probs[1] if prediction == 1 else probs[0]
    is_active = bool(prediction == 1)

    return {
        "success": True,
        "is_active": is_active,
        "label": "NR3C1 AGONIST" if is_active else "NON-BINDER",
        "message": generate_mechanism_report(is_active, confidence, name),
        "confidence": float(confidence),
        "mol_3d": mol_3d,
        "flow_active": is_active
    }

def analyze_interaction(drug_a, drug_b):
    """Dual Drug Analysis"""
    mol_3d_a = get_pubchem_3d(drug_a)
    mol_3d_b = get_pubchem_3d(drug_b)

    # Resolve Names Properly
    smiles_a, name_a = resolve_name(drug_a)
    smiles_b, name_b = resolve_name(drug_b)

    # 1. Check Expert Database
    input_set = {str(name_a).lower(), str(name_b).lower()}
    
    key_found = None
    for k in INTERACTION_DB:
        db_set = {x.lower() for x in k}
        # Check subset matches (handles "aspirin" vs "Aspirin")
        if db_set.issubset(input_set) or input_set.issubset(db_set):
            key_found = k
            break

    if key_found:
        data = INTERACTION_DB[key_found]
        return {
            "success": True,
            "mode": "INTERACTION",
            "title": data['title'],
            "risk": data['risk'],
            "message": data['mechanism'],
            "mol_3d_a": mol_3d_a,
            "mol_3d_b": mol_3d_b
        }

    # 2. AI Simulation (Fallback)
    # Check if we have valid names, otherwise use generic terms
    display_a = name_a if name_a else "Primary Agent"
    display_b = name_b if name_b else "Secondary Agent"

    model = load_model()
    feat_a = featurize_smiles(smiles_a)
    feat_b = featurize_smiles(smiles_b)
    
    pred_a = model.predict(feat_a)[0] if feat_a is not None else 0
    pred_b = model.predict(feat_b)[0] if feat_b is not None else 0

    if pred_a == 1 and pred_b == 1:
        msg = (
            f"SIMULATION RESULT: COMPETITIVE BINDING DETECTED\n"
            f"Both {display_a} and {display_b} are identified as NR3C1 Ligands. "
            f"They will compete for the same receptor binding pocket in the cytoplasm. "
            f"This may lead to displacement of the lower-affinity drug."
        )
        risk = "MODERATE (Competition)"
    else:
        msg = (
            f"SIMULATION RESULT: INDEPENDENT PATHWAYS\n"
            f"{display_a} and {display_b} appear to operate on distinct cellular targets. "
            f"No direct receptor competition or synergistic toxicity is predicted by the structural model."
        )
        risk = "LOW"

    return {
        "success": True,
        "mode": "SIMULATION",
        "title": "AI-GENERATED INTERACTION PROFILE",
        "risk": risk,
        "message": msg,
        "mol_3d_a": mol_3d_a,
        "mol_3d_b": mol_3d_b
    }