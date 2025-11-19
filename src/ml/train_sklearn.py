"""
Scikit-Learn Random Forest Training with MLflow
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
import shap

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


def train_sklearn_model():
    print("="*70)
    print("SKLEARN RANDOM FOREST TRAINING")
    print("="*70)

    # Load data
    data_path = PROJECT_ROOT / "data" / "transactions_raw.csv"
    df = pd.read_csv(data_path)

    # Features
    feature_cols = [
        'amount', 'is_online', 'hour', 'day_of_week', 'is_weekend', 'is_night',
        'time_since_last_transaction', 'customer_avg_amount', 'customer_std_amount',
        'amount_deviation', 'merchant_risk_score', 'distance_from_home'
    ]

    X = df[feature_cols].fillna(0).values
    y = df['is_fraud'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"✅ Original train: {len(X_train)}, Balanced: {len(X_train_balanced)}")

    # MLflow tracking
    mlflow.set_experiment("fraud_detection_sklearn")

    with mlflow.start_run():
        params = {
            "model_type": "random_forest",
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 10,
            "class_weight": "balanced"
        }
        mlflow.log_params(params)

        # Train
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        print("\n🚀 Training...")
        clf.fit(X_train_balanced, y_train_balanced)

        # Evaluate
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"\n🎯 ROC-AUC: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred))

        mlflow.log_metric("roc_auc", roc_auc)

        # Save model
        model_dir = PROJECT_ROOT / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "sklearn_rf_model.pkl"
        joblib.dump(clf, model_path)

        # Create SHAP explainer
        print("\n🔍 Creating SHAP explainer...")
        explainer = shap.TreeExplainer(clf)
        explainer_path = model_dir / "sklearn_rf_explainer.pkl"
        joblib.dump(explainer, explainer_path)

        mlflow.sklearn.log_model(clf, "model")

        print(f"\n✅ Model saved to {model_path}")
        print(f"✅ Explainer saved to {explainer_path}")

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    train_sklearn_model()
