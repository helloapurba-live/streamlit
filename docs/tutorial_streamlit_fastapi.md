# Complete Tutorial: Streamlit + FastAPI ML Dashboard

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Setup & Installation](#setup--installation)
4. [Data Generation](#data-generation)
5. [Model Training](#model-training)
6. [Backend API](#backend-api)
7. [Frontend Dashboard](#frontend-dashboard)
8. [MLOps Integration](#mlops-integration)
9. [Deployment](#deployment)

---

## Introduction

This tutorial teaches you how to build a **production-ready ML web application** for bank transaction fraud detection using:

- **Frontend**: Streamlit (dashboard-first UX)
- **Backend**: FastAPI (REST API with async batch processing)
- **ML Stack**: PyTorch + Skorch + Scikit-Learn + Optuna
- **MLOps**: MLflow tracking, feature store, model registry
- **Platform**: Windows 11, CPU-only, open-source Python

### Learning Path

1. **Beginner → Intermediate**: Understand each component individually
2. **Intermediate → Advanced**: Integrate components into a cohesive system
3. **Advanced**: Production hardening and MLOps best practices

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│                    http://localhost:8501                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STREAMLIT FRONTEND                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Overview │  │ Predict  │  │  Batch   │  │Investigate│       │
│  │  Page    │  │  Page    │  │  Page    │  │  Page    │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP Requests
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  POST /predict_single  → Real-time prediction            │  │
│  │  POST /predict_batch   → Async batch job submission      │  │
│  │  GET  /job_status/:id  → Poll job progress               │  │
│  │  GET  /download/:id    → Download batch results          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   Model      │    │  Job Queue   │    │   SQLite     │    │
│  │   Manager    │    │  Background  │    │   Job DB     │    │
│  │  (sklearn/   │    │   Tasks      │    │              │    │
│  │   PyTorch)   │    │              │    │              │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA & MODELS                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │    Data      │  │   Models     │  │   MLflow     │         │
│  │ transactions │  │  sklearn_rf  │  │  Tracking    │         │
│  │    .csv      │  │ pytorch_mlp  │  │   mlruns/    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Setup & Installation

### Prerequisites

- Windows 11 (64-bit)
- Python 3.10 or 3.11
- 8GB RAM minimum
- 5GB free disk space

### Step-by-Step Setup

#### 1. Create Project Directory

```powershell
# Create project folder
New-Item -Path "C:\ml-fraud-dashboard" -ItemType Directory
cd C:\ml-fraud-dashboard

# Download or clone the project files here
```

#### 2. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate (may require ExecutionPolicy change)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1

# Verify activation (should see (.venv) prefix)
```

#### 3. Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch CPU version (important for Windows)
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install all other packages
pip install -r requirements.txt

# Verify installation
python -c "import torch, streamlit, fastapi, skorch, optuna; print('✅ All imports successful!')"
```

**Troubleshooting**:
- If DLL errors occur, reinstall PyTorch with CPU flag
- If Streamlit won't start, clear cache: `streamlit cache clear`
- If port conflicts, change ports in scripts

---

## Data Generation

### Understanding the Synthetic Data

The fraud detection dataset requires:
- **Imbalanced classes**: 97% normal, 3% fraud
- **Temporal features**: hour, day of week, velocity
- **Behavioral features**: deviation from customer baseline
- **Network features**: merchant risk, location anomalies

### Generate Data

```powershell
python data\generate_synthetic_transactions.py
```

**Output**:
- `data/transactions_raw.csv` (10,000 rows)
- Fraud rate: ~3%
- 1,000 unique customers
- 21 features

### Data Schema

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | string | Unique transaction ID |
| `customer_id` | string | Customer identifier |
| `timestamp` | datetime | Transaction timestamp |
| `amount` | float | Transaction amount (USD) |
| `merchant_category` | string | Merchant type (grocery, etc.) |
| `merchant_id` | string | Merchant identifier |
| `latitude` | float | Transaction location (lat) |
| `longitude` | float | Transaction location (lon) |
| `is_online` | int | 0=in-person, 1=online |
| `is_fraud` | int | **Label**: 0=normal, 1=fraud |
| `hour` | int | Hour of day (0-23) |
| `day_of_week` | int | Day of week (0=Mon, 6=Sun) |
| `is_weekend` | int | 1 if Sat/Sun |
| `is_night` | int | 1 if 12am-6am |
| `time_since_last_transaction` | float | Minutes since last transaction |
| `customer_avg_amount` | float | Customer's average transaction |
| `customer_std_amount` | float | Customer's std deviation |
| `amount_deviation` | float | Z-score from baseline |
| `merchant_risk_score` | float | Merchant fraud rate |
| `distance_from_home` | float | Distance from customer home |

---

## Model Training

### Training Workflow

```powershell
# Run complete pipeline (recommended)
.\scripts\run_full_pipeline.ps1

# Or train individual models:
python src\ml\train_sklearn.py     # Random Forest
python src\ml\train_pytorch.py     # PyTorch MLP via Skorch
python src\ml\optuna_tune.py       # Hyperparameter tuning
```

### Scikit-Learn Random Forest

**File**: `src/ml/train_sklearn.py`

**Key Concepts**:
- Uses `imbalanced-learn` SMOTE for class balancing
- `class_weight='balanced'` for better minority class learning
- SHAP TreeExplainer for model interpretability

**Training Steps**:
1. Load data and split (80/20 train/test)
2. Apply SMOTE to balance training set
3. Train RandomForestClassifier (100 trees, max_depth=10)
4. Evaluate with ROC-AUC
5. Create SHAP explainer
6. Log to MLflow
7. Save model and explainer to `models/`

**Expected Results**:
- ROC-AUC: ~0.90-0.95
- Precision: ~0.85-0.92
- Recall: ~0.75-0.88

### PyTorch + Skorch MLP

**File**: `src/ml/train_pytorch.py`

**Key Concepts**:
- **Skorch wraps PyTorch models** to provide sklearn API
- Enables using PyTorch models in sklearn Pipelines
- Compatible with GridSearchCV, Optuna, etc.

**Architecture**:
```python
FraudDetectionMLP(
    Input(12) → Dense(64) → ReLU → Dropout(0.3) →
    Dense(32) → ReLU → Dropout(0.2) →
    Dense(16) → ReLU →
    Dense(2) → Softmax
)
```

**Skorch Integration**:
```python
from skorch import NeuralNetClassifier

net = NeuralNetClassifier(
    FraudDetectionMLP,        # PyTorch nn.Module
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    lr=0.001,
    batch_size=128,
    max_epochs=50,
    callbacks=[
        EarlyStopping(patience=10),
        EpochScoring('roc_auc')
    ]
)

# Now use like sklearn estimator!
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('net', net)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

**Why Skorch?**
- ✅ Consistent API across all models
- ✅ Easy hyperparameter tuning
- ✅ Integration with sklearn utilities
- ✅ Callbacks for training control

### Optuna Hyperparameter Tuning

**File**: `src/ml/optuna_tune.py`

**Key Concepts**:
- Bayesian optimization for hyperparameter search
- Works seamlessly with Skorch models
- Much faster than GridSearch for deep learning

**Tuned Parameters**:
- `hidden_dim1`: [32, 128]
- `hidden_dim2`: [16, 64]
- `hidden_dim3`: [8, 32]
- `dropout`: [0.1, 0.5]
- `lr`: [1e-4, 1e-2] (log scale)
- `batch_size`: [64, 128, 256]

**Optimization Process**:
```python
def objective(trial, X_train, y_train):
    # Suggest hyperparameters
    hidden_dim1 = trial.suggest_int('hidden_dim1', 32, 128)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    # Create model with suggested params
    net = NeuralNetClassifier(
        FraudDetectionMLP,
        module__hidden_dim1=hidden_dim1,
        lr=lr,
        ...
    )

    # Evaluate with cross-validation
    scores = cross_val_score(net, X_train, y_train, cv=3)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

**Results**:
- Best ROC-AUC logged to MLflow
- Best parameters saved for production use

---

## Backend API

### FastAPI Application Overview

**File**: `src/backend/app.py`

**Core Components**:
1. **Pydantic Models**: Request/response validation
2. **Model Manager**: Load and cache ML models
3. **Job Tracker**: SQLite database for batch jobs
4. **Background Tasks**: Async batch processing
5. **Prometheus Metrics**: Monitoring

### Endpoints

#### 1. POST /predict_single

**Purpose**: Real-time fraud prediction

**Request**:
```json
{
  "customer_id": "CUST_000123",
  "amount": 850.00,
  "merchant_category": "online_shopping",
  "merchant_id": "MERCH_09876",
  "latitude": 40.7128,
  "longitude": -74.0060,
  "is_online": 1,
  "hour": 23,
  "day_of_week": 5,
  "is_weekend": 1,
  "is_night": 1
}
```

**Response**:
```json
{
  "transaction_id": "uuid...",
  "customer_id": "CUST_000123",
  "fraud_probability": 0.78,
  "is_fraud_predicted": true,
  "model_version": "sklearn_rf",
  "explanation": {
    "method": "shap",
    "top_features": [
      {"feature": "amount", "shap_value": 0.25, "feature_value": 850.0}
    ]
  },
  "processing_time_ms": 15.3
}
```

#### 2. POST /predict_batch

**Purpose**: Async batch processing

**Request**: Multipart form with CSV file

**Response**:
```json
{
  "job_id": "batch_uuid-...",
  "status": "pending",
  "message": "Processing 1000 records",
  "submitted_at": "2025-01-19T10:30:00"
}
```

**Workflow**:
1. Upload CSV → job created in SQLite
2. Background task processes rows in batches of 100
3. Updates progress in database
4. Saves results to CSV when complete

#### 3. GET /job_status/{job_id}

**Purpose**: Poll batch job progress

**Response**:
```json
{
  "job_id": "batch_uuid-...",
  "status": "running",
  "progress": 45.5,
  "total_records": 1000,
  "processed_records": 455
}
```

**Status Values**: `pending`, `running`, `completed`, `failed`

#### 4. GET /download_result/{job_id}

**Purpose**: Download completed batch results

**Response**: CSV file with predictions

### Running the Backend

```powershell
# Terminal 1: Start backend
.\scripts\run_backend.ps1

# Access at:
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - Metrics: http://localhost:8000/metrics
```

---

## Frontend Dashboard

### Streamlit Multi-Page Application

**File**: `src/frontend/app.py`

### Pages

#### 1. 📊 Overview (Dashboard Home)

**Features**:
- Real-time metrics (transactions, fraud count, fraud rate)
- Fraud rate gauge chart
- Time series of fraud detections
- Recent high-risk transactions table
- Model performance summary

**UX Design**:
- Card-based layout for metrics
- Color coding (green=safe, yellow=warning, red=fraud)
- Auto-refresh capability

#### 2. 🔍 Single Prediction

**Features**:
- Form with transaction input fields
- Real-time prediction on submit
- SHAP explanation visualization (bar chart)
- Colored result cards (red=fraud, green=safe)

**User Flow**:
1. Enter transaction details
2. Click "Analyze Transaction"
3. API call to `/predict_single`
4. Display result with confidence and explanation
5. Show recommended action (BLOCK/APPROVE)

#### 3. 📦 Batch Processing

**Features**:
- CSV file uploader with preview
- Job submission
- Real-time progress polling (5-second intervals)
- Progress bar
- Result download button
- Summary statistics on results

**User Flow**:
1. Upload CSV → shows preview
2. Submit job → receives job_id
3. Polling loop starts (max 5 minutes)
4. Progress bar updates every 5 seconds
5. When complete, download button appears
6. Download results CSV with predictions

**Technical Details**:
- Uses `st.session_state` to persist job_id
- Polling implemented with `time.sleep(5)` in loop
- `st.empty()` placeholders for dynamic updates

#### 4. 🕵️ Investigation

**Features**:
- Customer ID search
- Transaction history table (mock data)
- LLM-powered investigation query
- Risk score display
- Key findings and recommended actions

**Demo Flow**:
1. Enter customer ID
2. System displays transaction history
3. Enter investigation query
4. Calls `/agent_investigate` endpoint
5. Shows AI-generated summary, risk score, findings, actions

#### 5. ⚙️ Admin

**Features**:
- System health checks (API, MLflow, Streamlit)
- Quick links (FastAPI docs, MLflow UI, Prometheus)
- Model information display
- Configuration settings
- Danger zone (reload models, clear jobs)

### Running the Frontend

```powershell
# Terminal 2: Start frontend (backend must be running)
.\scripts\run_frontend.ps1

# Access at: http://localhost:8501
```

---

## MLOps Integration

### MLflow Experiment Tracking

**Start MLflow UI**:
```powershell
# Terminal 3: Start MLflow
.\scripts\run_mlflow.ps1

# Access at: http://localhost:5000
```

**What's Tracked**:
- Parameters (model type, hyperparameters)
- Metrics (ROC-AUC, precision, recall)
- Artifacts (trained models, plots)
- System info (Python version, packages)

**View Experiments**:
1. Navigate to http://localhost:5000
2. Click on experiment name (e.g., "fraud_detection_sklearn")
3. View runs, compare metrics, download models

### Feature Store

**Location**: `feature_store/`

**Versioning**:
```
feature_store/
  features_v1_20250119.parquet
  features_v2_20250120.parquet
  feature_metadata.json
```

**Usage**:
```python
import pandas as pd

# Read latest features
df = pd.read_parquet('feature_store/features_v1_20250119.parquet')

# Write new version
df_new.to_parquet(f'feature_store/features_v2_{date}.parquet')
```

### Model Registry

**Location**: `models/`

**Structure**:
```
models/
  sklearn_rf_model.pkl
  sklearn_rf_explainer.pkl
  pytorch_mlp_model.pkl
  model_registry.json
```

**Registry Metadata** (`model_registry.json`):
```json
{
  "models": [
    {
      "name": "sklearn_rf",
      "version": "1.0",
      "stage": "production",
      "created_at": "2025-01-19T10:00:00",
      "metrics": {
        "roc_auc": 0.92,
        "precision": 0.88
      }
    }
  ]
}
```

---

## Deployment

### Local Development (3 Terminals)

```powershell
# Terminal 1: MLflow
.\scripts\run_mlflow.ps1

# Terminal 2: Backend
.\scripts\run_backend.ps1

# Terminal 3: Frontend
.\scripts\run_frontend.ps1
```

### Production Considerations

**Not covered in this tutorial but recommended**:

1. **Containerization**: Docker for backend + frontend
2. **Database**: PostgreSQL instead of SQLite
3. **Message Queue**: RabbitMQ/Redis for batch jobs
4. **Monitoring**: Full Prometheus + Grafana stack
5. **Authentication**: OAuth2 with JWT tokens
6. **HTTPS**: SSL certificates for secure communication
7. **Load Balancing**: Nginx reverse proxy
8. **Model Serving**: Separate model server (Triton/TorchServe)

---

## Next Steps

### Enhancements

1. **Add more fraud patterns** to data generator
2. **Implement model retraining pipeline** with Airflow/Prefect
3. **Add data quality checks** with Great Expectations
4. **Create model performance monitoring** dashboard
5. **Implement A/B testing** for model versions

### Further Learning

- **MLOps**: Read "Introducing MLOps" by Treveil et al.
- **FastAPI**: Official docs at fastapi.tiangolo.com
- **Streamlit**: Gallery at streamlit.io/gallery
- **Optuna**: Tutorials at optuna.readthedocs.io
- **Skorch**: Examples at skorch.readthedocs.io

---

## Troubleshooting

### Common Issues

**Issue**: "Port already in use"
```powershell
# Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Issue**: "MLflow database locked"
```powershell
# Stop MLflow and remove lock files
taskkill /IM mlflow.exe /F
Remove-Item mlruns\.mlflow.db-shm
Remove-Item mlruns\.mlflow.db-wal
```

**Issue**: "ImportError: DLL load failed"
```powershell
# Reinstall PyTorch CPU version
pip uninstall torch
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## Summary

You've built a complete ML web application with:

✅ Streamlit dashboard (5 pages, interactive UX)
✅ FastAPI backend (REST API, batch processing)
✅ PyTorch + Skorch + Optuna (deep learning + tuning)
✅ Scikit-Learn (classical ML baseline)
✅ MLflow tracking (experiment management)
✅ SHAP explanations (model interpretability)
✅ Background jobs (async batch predictions)
✅ Windows-compatible (PowerShell scripts, CPU-only)

**Congratulations! 🎉**
