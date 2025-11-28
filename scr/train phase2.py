import numpy as np
import pandas as pd
import os
import joblib
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# INPUTS: The "Math" files from Phase 2
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# OUTPUTS: Where we save the trained brain (the model)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_FILE = os.path.join(MODELS_DIR, 'stress_receptor_model.pkl')

def load_data():
    print("Loading training data...")
    try:
        X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print("❌ Error: Processed data not found. Run 'scr/preprocess_phase2.py' first.")
        sys.exit(1)

def train_ai():
    print("--- 🤖 Phase 3: Training the AI ---")
    
    # 1. Load the Math
    X_train, y_train, X_test, y_test = load_data()
    print(f"Training on {len(X_train)} compounds.")
    print(f"Testing on  {len(X_test)} compounds.")

    # 2. Initialize the Model (Random Forest)
    # n_estimators=100 means we create 100 decision trees
    # class_weight='balanced' helps because there are fewer Active drugs than Inactive ones
    print("\nInitializing Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, 
                                   max_depth=20, 
                                   random_state=42, 
                                   class_weight='balanced',
                                   n_jobs=-1) # Use all CPU cores

    # 3. Train (Fit)
    print("Training in progress (teaching the trees)...")
    model.fit(X_train, y_train)
    print("✅ Training Complete!")

    # 4. Evaluate (The Exam)
    print("\n--- 📊 Evaluation Results ---")
    
    # Ask the model to predict the Test Set
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] # Probability of being Active

    # Calculate Grades
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)

    print(f"Accuracy: {accuracy:.2%}")
    print(f"ROC-AUC:  {roc_auc:.4f} (1.0 is perfect, 0.5 is guessing)")
    
    print("\nConfusion Matrix:")
    # [Image of confusion matrix explanation]
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives (Correctly identified safe drugs): {cm[0][0]}")
    print(f"False Positives (Safe drugs flagged as active):   {cm[0][1]}")
    print(f"False Negatives (Active drugs missed):            {cm[1][0]}")
    print(f"True Positives (Correctly identified actives):    {cm[1][1]}")

    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=['Inactive', 'Active']))

    # 5. Save the Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\n💾 Model saved to: {MODEL_FILE}")
    print("   You can now use this file to predict new, unknown drugs!")

if __name__ == "__main__":
    train_ai()