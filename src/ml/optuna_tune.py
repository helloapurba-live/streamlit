"""
Optuna Hyperparameter Tuning for Skorch Models
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
import optuna
from optuna.integration import SkoptSampler
import mlflow

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ml.train_pytorch import FraudDetectionMLP


def objective(trial, X_train, y_train):
    """Optuna objective function for tuning Skorch model."""

    # Suggest hyperparameters
    hidden_dim1 = trial.suggest_int('hidden_dim1', 32, 128)
    hidden_dim2 = trial.suggest_int('hidden_dim2', 16, 64)
    hidden_dim3 = trial.suggest_int('hidden_dim3', 8, 32)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    # Create model
    net = NeuralNetClassifier(
        FraudDetectionMLP,
        module__input_dim=12,
        module__hidden_dim1=hidden_dim1,
        module__hidden_dim2=hidden_dim2,
        module__hidden_dim3=hidden_dim3,
        module__dropout=dropout,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=lr,
        batch_size=batch_size,
        max_epochs=20,  # Reduced for tuning speed
        device='cpu',
        verbose=0
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('net', net)
    ])

    # Cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='roc_auc')
    return scores.mean()


def tune_model():
    print("="*70)
    print("OPTUNA HYPERPARAMETER TUNING")
    print("="*70)

    # Load data
    data_path = PROJECT_ROOT / "data" / "transactions_raw.csv"
    df = pd.read_csv(data_path)

    feature_cols = [
        'amount', 'is_online', 'hour', 'day_of_week', 'is_weekend', 'is_night',
        'time_since_last_transaction', 'customer_avg_amount', 'customer_std_amount',
        'amount_deviation', 'merchant_risk_score', 'distance_from_home'
    ]

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df['is_fraud'].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✅ Data loaded: {len(X_train)} train samples")

    # Create Optuna study
    study = optuna.create_study(direction='maximize')

    print("\n🔍 Starting optimization (10 trials)...")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=10,
        show_progress_bar=True
    )

    print(f"\n🎯 Best ROC-AUC: {study.best_value:.4f}")
    print(f"📊 Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Log to MLflow
    mlflow.set_experiment("fraud_detection_optuna")
    with mlflow.start_run():
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_roc_auc", study.best_value)

    print("\n✅ Tuning complete!")


if __name__ == "__main__":
    tune_model()
