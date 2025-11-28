import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import sys

# --- ROBUST PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed_data.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'dexa_interaction_model.pkl')

def train():
    print("--- 🧠 Training Started ---")

    # 1. Robust Data Check
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File not found at {DATA_PATH}")
        print("   -> Run 'python scr/preprocess.py' first.")
        return

    # 2. Load Data with Error Handling
    print(f"   Loading data from: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
        if df.empty:
            raise ValueError("File exists but contains no data.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR loading data: {e}")
        print("   -> Your processed_data.csv is corrupted or empty.")
        print("   -> Please run 'python scr/preprocess.py' again to regenerate it.")
        return

    # 3. Prepare Features
    # The last column is 'Target'
    try:
        X = df.iloc[:, :-1].values
        y = df['Target'].values
    except KeyError:
        print("❌ Error: Data format invalid. Re-run preprocess.py")
        return
    
    print(f"   Training on {len(X)} compounds with {X.shape[1]-1} features each.")
    
    # 4. Split
    # stratify=y ensures we have a fair mix of active/inactive in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Initialize Model
    # class_weight='balanced' helps the AI find the rare positive cases
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,  # Use all CPUs
        class_weight='balanced',
        random_state=42
    )
    
    # 6. Fit
    print("   Fitting model (this may take 1-2 minutes)...")
    clf.fit(X_train, y_train)
    
    # 7. Evaluate
    print("\n--- 📊 Validation Results ---")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=['No Interaction', 'Interacts']))
    
    # 8. Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train()