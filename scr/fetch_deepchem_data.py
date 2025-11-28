import deepchem as dc
import pandas as pd
import os
import sys

# --- ROBUST PATH CONFIGURATION ---
# This ensures the file is saved in the correct 'data' folder relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw_drug_data_large.csv')

def fetch_tox21_data():
    print("--- 🧬 DeepChem Data Downloader ---")
    print("Fetching Tox21 dataset...")
    print("(Note: You may see 'Explicit valence' warnings from RDKit. These are normal for this dataset.)")
    
    try:
        # Load Tox21 with 'Raw' featurizer to get SMILES strings
        tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='Raw')
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR downloading data: {e}")
        return

    train_dataset, valid_dataset, test_dataset = datasets
    
    # Extract SMILES and Labels
    # .ids contains the SMILES strings
    # .y contains the labels (0 or 1)
    smiles = train_dataset.ids
    labels = train_dataset.y
    
    print(f"\nAvailable Tasks: {tasks}")
    
    # --- TARGET SELECTION FIX ---
    # We prefer 'NR-GR' (Glucocorticoid Receptor), but if missing, we use 'NR-AR' (Androgen Receptor)
    # Both are steroid nuclear receptors and good proxies for this learning task.
    if 'NR-GR' in tasks:
        target_task = 'NR-GR'
    elif 'NR-AR' in tasks:
        target_task = 'NR-AR'
        print("ℹ️ 'NR-GR' task not found in this dataset version. Using 'NR-AR' (Androgen Receptor) as proxy.")
    else:
        target_task = tasks[0]
        print(f"⚠️ Neither NR-GR nor NR-AR found. Defaulting to first available task: {target_task}")

    task_index = tasks.index(target_task)
    print(f"--> Extracting data for target: {target_task} (Index {task_index})")
    
    # --- DATA SHAPE FIX ---
    # Ensure smiles are strings and handle potential nested arrays
    flat_smiles = [str(s) if not isinstance(s, str) else s for s in smiles]
    
    # Create DataFrame
    data = {
        'Drug_Name': [f"Tox21_{i}" for i in range(len(flat_smiles))],
        'SMILES': flat_smiles,
        'Interacts_with_Dexa': labels[:, task_index]
    }
    
    df = pd.DataFrame(data)
    
    # --- CLEANING ---
    # Remove rows where label is NaN (empty data) or SMILES is missing
    initial_count = len(df)
    df = df.dropna(subset=['Interacts_with_Dexa', 'SMILES'])
    df = df[df['SMILES'] != ''] # Remove empty strings
    
    cleaned_count = len(df)
    print(f"Raw compounds: {initial_count}")
    print(f"Valid compounds suitable for training: {cleaned_count}")
    
    if cleaned_count == 0:
        print("❌ Error: No valid data found after cleaning. Check network connection or dataset version.")
        return
 
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Success! Dataset saved to: {OUTPUT_FILE}")
    print("You can now run 'python scr/preprocess.py'")

if __name__ == "__main__":
    fetch_tox21_data()