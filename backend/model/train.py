import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

def train_failure_model(data_path='../data/processed/synthetic_training_data.csv'):
    # 1. Load Data
    if not os.path.exists(data_path):
        print("Data not found. Please run generate_mock_data.py first.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records.")

    # 2. Preprocessing
    feature_cols = [
        'rain_1h', 'rain_3h', 'rain_forecast_3h', 'antecedent_rain_24h',
        'mean_elevation', 'drain_density', 'low_lying_pct',
        'pothole_severity_index'
    ]
    target_col = 'failure_event'

    # Drop NA if any
    df = df.dropna()

    X = df[feature_cols]
    y = df[target_col]

    # Time-based split (ideal) or random for now
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

    # 3. Model Training
    # Class weights balanced because floods are rare events
    model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Performance ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # 5. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n--- Feature Importances ---")
    for f in range(X.shape[1]):
        print(f"{feature_cols[indices[f]]}: {importances[indices[f]]:.4f}")

    # Save artifact
    joblib.dump(model, 'ward_failure_model_v1.pkl')
    print("\nModel saved to ward_failure_model_v1.pkl")

if __name__ == "__main__":
    train_failure_model()
