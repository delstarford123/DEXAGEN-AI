import pandas as pd
import os
import sys
from rdkit import Chem

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw_drug_dataset_large.csv')

# Direct URL to the Tox21 dataset used by DeepChem
TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

def is_valid_molecule(smiles):
    """
    Checks if a SMILES string represents a valid molecule using RDKit.
    Returns True if valid, False if broken (like the Aluminum errors).
    """
    if not isinstance(smiles, str) or smiles == "":
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def fetch_tox21_data_manual():
    print("--- 🧪 Manual Data Downloader ---")
    print(f"Downloading raw CSV directly from: {TOX21_URL}")
    
    try:
        # 1. Download directly using Pandas (Bypassing DeepChem loader)
        df_raw = pd.read_csv(TOX21_URL, compression='gzip')
    except Exception as e:
        print(f"\n❌ Network Error: Could not download file. {e}")
        return

    print(f"Original Raw Data Size: {len(df_raw)} rows")
    
    # 2. Identify the Tasks (Columns)
    available_columns = df_raw.columns.tolist()
    print(f"\nAvailable Columns: {available_columns}")
    
    # 3. Target Selection Logic
    if 'NR-GR' in available_columns:
        target_task = 'NR-GR'
    elif 'NR-AR' in available_columns:
        target_task = 'NR-AR'
        print("ℹ️ 'NR-GR' (Glucocorticoid) not found in header. Using 'NR-AR' (Androgen) as proxy.")
    else:
        # Fallback to the first valid task column (usually index 0-11 are tasks)
        target_task = available_columns[0]
        print(f"⚠️ Target not found. Defaulting to: {target_task}")

    print(f"--> Extracting data for target: {target_task}")

    # 4. Filter and Clean
    # We create a new list of valid data to ensure no "garbage" crashes us
    valid_data = []
    
    print("Cleaning data (removing broken molecules)...")
    for index, row in df_raw.iterrows():
        smiles = row['smiles']
        label = row[target_task]
        
        # Check 1: Is the label a valid number? (Not NaN)
        if pd.isna(label):
            continue
            
        # Check 2: Is the chemistry valid?
        if is_valid_molecule(smiles):
            valid_data.append({
                'Drug_Name': f"Tox21_{row['mol_id']}",
                'SMILES': smiles,
                'Interacts_with_Receptor': label
            })

    # 5. Create Clean DataFrame
    df_clean = pd.DataFrame(valid_data)
    
    # Final check
    if len(df_clean) == 0:
        print("❌ Error: No valid data remained after cleaning.")
        return

    # 6. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Success! Dataset saved to: {OUTPUT_FILE}")
    print(f"   Original Rows: {len(df_raw)}")
    print(f"   Clean Rows:    {len(df_clean)}")
    print("   You can now run 'python scr/enrich_data.py'")

if __name__ == "__main__":
    fetch_tox21_data_manual()