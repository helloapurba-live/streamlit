"""
============================================================================
PyTorch + Skorch Model Training for Fraud Detection
============================================================================

LEARNING OBJECTIVES:
1. Wrap PyTorch nn.Module with Skorch for sklearn compatibility
2. Use Skorch models in sklearn Pipelines
3. Train deep learning model with proper callbacks
4. Log experiments to MLflow
5. Save model for inference

THEORY:
Skorch bridges PyTorch and scikit-learn by wrapping PyTorch modules
in estimators that follow sklearn's fit/predict API. This enables:
- Using PyTorch models in sklearn Pipelines
- Grid/RandomSearch compatibility
- Integration with Optuna
- Consistent API across classical ML and deep learning
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.sklearn
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, EarlyStopping, LRScheduler
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


class FraudDetectionMLP(nn.Module):
    """
    Multi-Layer Perceptron for fraud detection.

    Architecture:
    Input (12 features) → Dense(64) → ReLU → Dropout(0.3) →
    Dense(32) → ReLU → Dropout(0.2) → Dense(16) → ReLU →
    Dense(2, softmax) → Output (fraud/normal)
    """

    def __init__(self, input_dim=12, hidden_dim1=64, hidden_dim2=32,
                 hidden_dim3=16, dropout=0.3):
        super(FraudDetectionMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(dropout * 0.7)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, 2)  # Binary classification

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # Return logits (not softmax) for CrossEntropyLoss
        # Skorch applies softmax internally in predict_proba()
        return x


def train_pytorch_model():
    """Train PyTorch model wrapped with Skorch."""

    print("="*70)
    print("PYTORCH + SKORCH FRAUD DETECTION MODEL TRAINING")
    print("="*70)

    # Load data
    data_path = PROJECT_ROOT / "data" / "transactions_raw.csv"
    print(f"\n📂 Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Prepare features
    feature_cols = [
        'amount', 'is_online', 'hour', 'day_of_week', 'is_weekend', 'is_night',
        'time_since_last_transaction', 'customer_avg_amount', 'customer_std_amount',
        'amount_deviation', 'merchant_risk_score', 'distance_from_home'
    ]

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df['is_fraud'].values.astype(np.int64)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✅ Train set: {len(X_train)} samples")
    print(f"✅ Test set: {len(X_test)} samples")
    print(f"✅ Fraud rate: {y.mean()*100:.2f}%")

    # Start MLflow run
    mlflow.set_experiment("fraud_detection_pytorch")

    with mlflow.start_run():
        # Log parameters
        params = {
            "model_type": "pytorch_mlp",
            "hidden_dim1": 64,
            "hidden_dim2": 32,
            "hidden_dim3": 16,
            "dropout": 0.3,
            "lr": 0.001,
            "batch_size": 128,
            "max_epochs": 50
        }
        mlflow.log_params(params)

        # Create Skorch classifier
        net = NeuralNetClassifier(
            FraudDetectionMLP,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            lr=0.001,
            batch_size=128,
            max_epochs=50,
            callbacks=[
                EarlyStopping(patience=10),
                EpochScoring('roc_auc', lower_is_better=False),
            ],
            device='cpu',  # Use 'cuda' if GPU available
            verbose=1
        )

        # Create pipeline with StandardScaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('net', net)
        ])

        # Train model
        print("\n🚀 Training model...")
        pipeline.fit(X_train, y_train)

        # Evaluate
        print("\n📊 Evaluating model...")
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        # Metrics
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        print(f"\n🎯 ROC-AUC: {roc_auc:.4f}")
        print("\n" + classification_report(y_test, y_pred))

        # Log metrics
        mlflow.log_metric("roc_auc", roc_auc)

        # Save model
        model_dir = PROJECT_ROOT / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "pytorch_mlp_model.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"\n✅ Model saved to {model_path}")

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    train_pytorch_model()
