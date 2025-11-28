import pandas as pd
import numpy as np
import deepchem as dc
import os
import sys
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# INPUT: The file you created in the previous step
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'personalized_drug_data.csv')

# OUTPUTS: Where we save the "Math" ready for AI training
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

def preprocess_for_ai():
    print("--- 🧠 Phase 2: Preprocessing & Featurization ---")
    
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Could not find {INPUT_FILE}")
        return

    print("Loading personalized dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} compounds.")

    # 2. Featurization (Text -> Math)
    # We use Extended Connectivity Fingerprints (ECFP).
    # This turns every molecule into a list of 1024 numbers (0s and 1s).
    print("\nEncoding chemicals into ECFP4 Fingerprints...")
    print("(This translates the chemical structure into binary code)")
    
    featurizer = dc.feat.CircularFingerprint(size=1024, radius=2)
    
    # DeepChem converts the SMILES column into numpy arrays
    features = featurizer.featurize(df['SMILES'].tolist())
    
    # The 'y' (Target) is whether it interacts with the receptor
    labels = df['Interacts_with_Receptor'].to_numpy()
    
    # 3. Clean Failures
    # Sometimes featurization fails on weird molecules. We remove them.
    # Check for empty arrays in features
    valid_indices = []
    for i, feat in enumerate(features):
        if feat.shape == (1024,) and not np.isnan(labels[i]):
            valid_indices.append(i)
            
    print(f"Original Count: {len(df)}")
    print(f"Valid Count:    {len(valid_indices)}")
    
    # Filter data
    X = np.stack([features[i] for i in valid_indices])
    y = labels[valid_indices]
    
    # Keep track of names for the results later
    names = df.iloc[valid_indices]['Drug_Name'].tolist()

    # 4. Split Data (Training vs Testing)
    # We hide 20% of the data to test the AI later (like a final exam)
    # stratify=y ensures we have a mix of Active/Inactive in both sets
    print("\nSplitting data into Training (80%) and Test (20%) sets...")
    
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Save to Disk
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)
    
    # Save names as CSVs so we can look them up later
    pd.DataFrame(names_train, columns=['Drug_Name']).to_csv(
        os.path.join(PROCESSED_DIR, 'train_names.csv'), index=False
    )
    pd.DataFrame(names_test, columns=['Drug_Name']).to_csv(
        os.path.join(PROCESSED_DIR, 'test_names.csv'), index=False
    )

    print(f"\n✅ Phase 2 Complete!")
    print(f"   Training Data: {X_train.shape[0]} drugs")
    print(f"   Test Data:     {X_test.shape[0]} drugs")
    print(f"   Files saved to: {PROCESSED_DIR}")
    
    # Quick Check on Manual Drugs
    print("\nWhere did your manual drugs go?")
    for name in ['DEXAMETHASONE_MANUAL', 'ASPIRIN_MANUAL', 'CORTISOL_MANUAL']:
        if name in names_train:
            print(f"   - {name} -> Training Set")
        elif name in names_test:
            print(f"   - {name} -> Test Set (The AI will be tested on this!)")

if __name__ == "__main__":
    preprocess_for_ai()