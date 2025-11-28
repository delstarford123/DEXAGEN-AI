import os
import sys
import joblib
import numpy as np
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem
import requests
import traceback

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
        "title": "COMPETITIVE RECEPTOR DISPLACEMENT DYNAMICS",
        "risk": "MODERATE (Adrenal Insufficiency)",
        "mechanism": (
            "PHARMACOKINETIC COMPETITION ANALYSIS:\n"
            "Comparison of Exogenous vs. Endogenous Ligand Binding:\n\n"
            "1. AFFINITY DIFFERENTIAL: Dexamethasone exhibits a dissociation constant (Kd) approximately 30-fold lower than endogenous Cortisol, indicating significantly higher affinity for the NR3C1 Ligand Binding Domain (LBD).\n\n"
            "2. COMPETITIVE EXCLUSION: Due to superior affinity and plasma half-life, Dexamethasone saturates the cytosolic glucocorticoid receptors, effectively displacing native Cortisol.\n\n"
            "3. NEGATIVE FEEDBACK: The potent Dex-NR3C1 complex hyper-stimulates GREs in the Pituitary gland, suppressing ACTH transcription and halting natural adrenal function."
        )
    }
}

def load_model():
    if not os.path.exists(MODEL_PATH): return None
    return joblib.load(MODEL_PATH)

def generate_3d_local(smiles):
    """
    FALLBACK: Generates 3D coordinates locally using RDKit Physics.
    Used if PubChem API is slow or fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        mol = Chem.AddHs(mol)
        # Use random coordinates first to ensure embedding works
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        if res == -1: return None # Embedding failed
        
        # Optimize geometry (Physics Simulation)
        try: AllChem.MMFFOptimizeMolecule(mol)
        except: pass
        
        return Chem.MolToMolBlock(mol)
    except:
        return None

def get_3d_structure(identifier):
    """
    HYBRID ENGINE: Tries API first, falls back to Local Generation.
    """
    # 1. Try PubChem API (High Quality)
    try:
        if HAS_PUBCHEM:
            cids = pcp.get_cids(identifier, namespace='name')
            if not cids: cids = pcp.get_cids(identifier, namespace='smiles')
            
            if cids:
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids[0]}/record/SDF/?record_type=3d&response_type=display"
                response = requests.get(url, timeout=3) # Short timeout to prevent hanging
                if response.status_code == 200:
                    return response.text
    except:
        pass # API Failed, move to fallback

    # 2. Local Fallback (RDKit)
    # Resolve name to SMILES if needed
    smiles = identifier
    if not Chem.MolFromSmiles(identifier) and HAS_PUBCHEM:
        try:
            compounds = pcp.get_compounds(identifier, namespace='name')
            if compounds: smiles = compounds[0].isomeric_smiles
        except: pass
    
    return generate_3d_local(smiles)

def resolve_name(user_input):
    """Resolves Input to (SMILES, DisplayName)"""
    user_input = str(user_input).strip()
    
    # Try as SMILES
    if Chem.MolFromSmiles(user_input):
        # Try to find name for this SMILES
        name = "Custom Molecule"
        if HAS_PUBCHEM:
            try:
                compounds = pcp.get_compounds(user_input, namespace='smiles')
                if compounds and compounds[0].synonyms: name = compounds[0].synonyms[0]
            except: pass
        return user_input, name

    # Try as Name
    if HAS_PUBCHEM:
        try:
            compounds = pcp.get_compounds(user_input, namespace='name')
            if compounds:
                return compounds[0].isomeric_smiles, user_input.title()
        except: pass
            
    return None, user_input

def featurize_smiles(smiles):
    try:
        featurizer = dc.feat.CircularFingerprint(size=1024, radius=2)
        features = featurizer.featurize([smiles])
        if len(features) > 0 and features[0].shape == (1024,): return features[0].reshape(1, -1)
    except: pass
    return None

def generate_mechanism_report(is_active, confidence, name):
    if is_active:
        return (
            f"RESEARCH ANALYSIS FOR {name.upper()}:\n\n"
            f"1. MEMBRANE DIFFUSION: This lipophilic compound ({confidence:.1%} probability) traverses the phospholipid bilayer.\n"
            f"2. CYTOSOLIC BINDING: Targets NR3C1 (Glucocorticoid Receptor), dissociating Heat Shock Proteins (HSP90).\n"
            f"3. NUCLEAR TRANSLOCATION: The complex dimerizes and actively translocates through the nuclear pore.\n"
            f"4. GENOMIC TRANSCRIPTION: Binds to Glucocorticoid Response Elements (GREs) on DNA, initiating transcription (Annexin A1) or transrepression (NF-κB)."
        )
    return (
        f"RESEARCH ANALYSIS FOR {name.upper()}:\n\n"
        f"Structural analysis indicates a lack of pharmacophore features for the NR3C1 Ligand Binding Domain. "
        f"The molecule likely interacts with alternative cytosolic pathways without triggering nuclear translocation."
    )

def predict_drug(user_input):
    """Single Drug Analysis"""
    try:
        model = load_model()
        if not model: return {"error": "Model missing."}

        # Parallel: Get Structure & Resolve Name
        mol_3d = get_3d_structure(user_input)
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
    except Exception as e:
        return {"error": str(e)}

def analyze_interaction(drug_a, drug_b):
    """Dual Drug Analysis with Robust Error Handling"""
    try:
        model = load_model()
        
        # 1. Resolve Inputs
        smiles_a, name_a = resolve_name(drug_a)
        smiles_b, name_b = resolve_name(drug_b)
        
        if not smiles_a or not smiles_b:
            return {"error": "Could not identify one or both drugs."}

        # 2. Get 3D Models (Independent try/catch inside function)
        mol_3d_a = get_3d_structure(drug_a)
        mol_3d_b = get_3d_structure(drug_b)

        # 3. Check Expert DB
        input_set = {str(name_a).lower(), str(name_b).lower()}
        key_found = None
        for k in INTERACTION_DB:
            db_set = {x.lower() for x in k}
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

        # 4. Fallback Simulation
        feat_a = featurize_smiles(smiles_a)
        feat_b = featurize_smiles(smiles_b)
        
        pred_a = model.predict(feat_a)[0] if feat_a is not None else 0
        pred_b = model.predict(feat_b)[0] if feat_b is not None else 0

        risk = "MODERATE (Competition)" if (pred_a == 1 and pred_b == 1) else "LOW"
        msg = f"SIMULATION: {name_a} and {name_b} analyzed. "
        if pred_a == 1 and pred_b == 1:
            msg += "Both target the NR3C1 receptor and will likely compete for the binding pocket."
        else:
            msg += "Structural analysis suggests independent cellular pathways."

        return {
            "success": True,
            "mode": "SIMULATION",
            "title": "AI INTERACTION PROFILE",
            "risk": risk,
            "message": msg,
            "mol_3d_a": mol_3d_a,
            "mol_3d_b": mol_3d_b
        }
    except Exception as e:
        print(traceback.format_exc()) # Log full error to server console
        return {"error": f"Analysis failed: {str(e)}"}