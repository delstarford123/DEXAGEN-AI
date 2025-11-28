import pandas as pd
import os
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# [CRITICAL] Matches the output filename from your fetch script logs
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw_drug_dataset_large.csv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'personalized_drug_data.csv')

def calculate_properties(smiles):
    """
    Calculates Molecular Weight and LogP (Solubility) using RDKit.
    Returns: (Weight, LogP, H-Bond Donors)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Molecular Weight: Heavier drugs usually have harder times crossing membranes
            mw = Descriptors.MolWt(mol)
            # LogP: Lipophilicity (Fat-loving vs Water-loving)
            logp = Descriptors.MolLogP(mol)
            # H-Bond Donors: Affects receptor binding
            hbd = Descriptors.NumHDonors(mol)
            return mw, logp, hbd
        else:
            return None, None, None
    except:
        return None, None, None

def inject_personal_drugs(df):
    """
    Manually injects a library of known drugs (Stress, Pain, Hormones).
    Note: The bulk Tox21 data uses IDs (e.g. Tox21_12345) because converting 
    7000+ chemical strings to English names requires an external API/Database.
    """
    print("\n💉 Injecting named reference drugs (Stress, Pain, Hormones)...")
    
    my_drugs = [
        # --- STRESS & INFLAMMATION (Glucocorticoids) ---
        {
            'Drug_Name': 'DEXAMETHASONE_MANUAL', 
            'SMILES': 'C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO',
            'Interacts_with_Receptor': 1.0, 
            'Notes': 'Synthetic Glucocorticoid (Potent Stress Mimic)'
        },
        {
            'Drug_Name': 'CORTISOL_MANUAL', 
            'SMILES': 'C[C@]12CCC(=O)C=C1CC[C@@H]3[C@@H]2[C@H](C[C@]4([C@H]3CC[C@@]4(C(=O)CO)O)C)O',
            'Interacts_with_Receptor': 1.0, 
            'Notes': 'Natural Stress Hormone (Hydrocortisone)'
        },
        {
            'Drug_Name': 'PREDNISONE_MANUAL', 
            'SMILES': 'C[C@]12CC(=O)C3C(CCC4=CC(=O)C=CC43C)C1CC(C2(C(=O)CO)O)O',
            'Interacts_with_Receptor': 1.0, 
            'Notes': 'Common Anti-inflammatory Steroid'
        },

        # --- PAIN & FEVER (NSAIDs/Analgesics) ---
        {
            'Drug_Name': 'ASPIRIN_MANUAL', 
            'SMILES': 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'Interacts_with_Receptor': 0.0, 
            'Notes': 'COX Inhibitor (Pain/Inflammation)'
        },
        {
            'Drug_Name': 'IBUPROFEN_MANUAL', 
            'SMILES': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'Interacts_with_Receptor': 0.0, 
            'Notes': 'NSAID Pain Reliever'
        },
        {
            'Drug_Name': 'ACETAMINOPHEN_MANUAL', 
            'SMILES': 'CC(=O)NC1=CC=C(O)C=C1',
            'Interacts_with_Receptor': 0.0, 
            'Notes': 'Paracetamol (Pain/Fever)'
        },

        # --- HORMONES (Controls) ---
        {
            'Drug_Name': 'TESTOSTERONE_MANUAL', 
            'SMILES': 'C[C@]12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O',
            'Interacts_with_Receptor': 0.0, # Binds to AR, not GR primarily
            'Notes': 'Male Sex Hormone'
        },
        {
            'Drug_Name': 'PROGESTERONE_MANUAL', 
            'SMILES': 'CC(=O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C',
            'Interacts_with_Receptor': 0.0, 
            'Notes': 'Pregnancy Hormone'
        },
        
        # --- MENTAL STRESS/ANXIETY ---
        {
            'Drug_Name': 'DIAZEPAM_MANUAL', 
            'SMILES': 'CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3',
            'Interacts_with_Receptor': 0.0, 
            'Notes': 'Valium (Anxiety/Stress Relief)'
        },
        {
            'Drug_Name': 'FLUOXETINE_MANUAL', 
            'SMILES': 'CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F',
            'Interacts_with_Receptor': 0.0, 
            'Notes': 'Prozac (Antidepressant)'
        }
    ]
    
    new_rows = pd.DataFrame(my_drugs)
    # Add new drugs to the top of the dataframe
    updated_df = pd.concat([new_rows, df], ignore_index=True)
    return updated_df

def process_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found at {INPUT_FILE}")
        print("   Please check the filename in your 'data' folder.")
        return

    print(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Inject Manual Drugs
    df = inject_personal_drugs(df)
    
    # 2. Enrich with Chemical Properties
    print("⚗️  Calculating chemical properties (Molecular Weight, LogP)...")
    print("    (This analyzes the structure of every drug - might take 10-20 seconds)")
    
    mws, logps, hbds = [], [], []
    
    for smile in df['SMILES']:
        mw, logp, hbd = calculate_properties(smile)
        mws.append(mw)
        logps.append(logp)
        hbds.append(hbd)
        
    df['Molecular_Weight'] = mws
    df['LogP'] = logps
    df['H_Bond_Donors'] = hbds
    
    # 3. Clean up failed calculations
    df = df.dropna(subset=['Molecular_Weight'])
    
    # 4. Tag Heavy Molecules (Lipinski Rule > 500 daltons)
    # Dexamethasone is near this limit, Aspirin is well below it.
    df['Is_Heavy_Molecule'] = df['Molecular_Weight'] > 500
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Personalized dataset saved to: {OUTPUT_FILE}")
    print(f"   Total Compounds: {len(df)}")
    print("   Added Dexamethasone, Aspirin, and Cortisol successfully.")

if __name__ == "__main__":
    process_dataset()