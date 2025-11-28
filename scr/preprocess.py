import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import os
import sys

# --- ROBUST PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw_drug_data_large.csv')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed_data.csv')

# Initialize Generator (New RDKit API to fix warnings)
# Radius 2 is standard for drug discovery (ECFP4)
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fingerprint(smiles):
    """
    Generates a molecular fingerprint using the modern RDKit Generator API.
    Returns a numpy array of 0s and 1s.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Fast NumPy generation
            return MORGAN_GEN.GetFingerprintAsNumPy(mol)
        return None
    except:
        return None

def main():
    print("--- 🧪 Preprocessing Started ---")
    
    # 1. Check Input File
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ CRITICAL ERROR: Raw data file not found at: {RAW_DATA_PATH}")
        print("   -> Please run 'python scr/fetch_deepchem_data.py' first.")
        sys.exit(1)
        
    # 2. Load Data
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"   Loaded raw file with {len(df)} rows.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not read CSV. {e}")
        sys.exit(1)

    if len(df) == 0:
        print("❌ CRITICAL ERROR: Raw CSV is empty. Run the fetch script again.")
        sys.exit(1)

    # 3. Vectorize
    print("   Vectorizing compounds (this takes 10-20 seconds)...")
    # Apply vectorization
    fingerprints = df['SMILES'].apply(smiles_to_fingerprint)
    
    # 4. Filter Invalid Molecules
    valid_mask = fingerprints.notnull()
    valid_fingerprints = fingerprints[valid_mask]
    valid_targets = df.loc[valid_mask, 'Interacts_with_Dexa']
    
    count = len(valid_fingerprints)
    print(f"   Valid molecules processed: {count}")
    
    if count == 0:
        print("❌ Error: No valid SMILES strings found. Check raw data format.")
        sys.exit(1)

    # 5. Create Feature Matrix
    print("   Creating feature matrix...")
    # stack ensures we get a proper 2D numpy array
    X = np.stack(valid_fingerprints.values)
    
    # 6. Save safely
    try:
        # Combine features and target into one DataFrame for saving
        # We use int8 to save disk space (0/1 don't need 64 bits)
        X_df = pd.DataFrame(X, dtype=np.int8)
        X_df['Target'] = valid_targets.values
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        
        X_df.to_csv(PROCESSED_DATA_PATH, index=False)
        
        # Verify file size
        file_size = os.path.getsize(PROCESSED_DATA_PATH)
        print(f"✅ Success! Data saved to: {PROCESSED_DATA_PATH}")
        print(f"   File Size: {file_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ Failed to write file: {e}")

if __name__ == "__main__":
    main()