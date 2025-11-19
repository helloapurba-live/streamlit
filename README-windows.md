# 🏦 Bank Transaction Anomaly Detection Dashboard
## Production ML Web App | Windows 11 Setup Guide

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Environment Setup](#environment-setup)
4. [Data Generation](#data-generation)
5. [Model Training](#model-training)
6. [Running the Application](#running-the-application)
7. [Testing](#testing)
8. [MLOps Workflows](#mlops-workflows)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

**System Requirements:**
- Windows 11 (64-bit)
- Python 3.10 or 3.11 (3.12 may have compatibility issues)
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space

**Install Python:**
```powershell
# Download from python.org and verify installation
python --version  # Should show Python 3.10.x or 3.11.x
```

---

## Project Structure

```
C:\Users\YourName\ml-fraud-dashboard\
│
├── data\                          # Raw and processed datasets
│   ├── generate_synthetic_transactions.py
│   ├── transactions_raw.csv
│   └── transactions_processed.parquet
│
├── feature_store\                 # Versioned feature datasets
│   ├── features_v1_20250119.parquet
│   └── feature_metadata.json
│
├── models\                        # Saved model artifacts
│   ├── sklearn_rf_v1\
│   ├── pytorch_mlp_v1\
│   └── model_registry.json
│
├── mlruns\                        # MLflow tracking database
│
├── src\
│   ├── backend\
│   │   └── app.py                # FastAPI application
│   ├── frontend\
│   │   └── app.py                # Streamlit dashboard
│   ├── ml\
│   │   ├── train_sklearn.py      # Classical ML training
│   │   ├── train_pytorch.py      # PyTorch + Skorch training
│   │   ├── optuna_tune.py        # Hyperparameter optimization
│   │   ├── predict.py            # Inference engine
│   │   └── feature_engineering.py
│   └── utils\
│       ├── config.py
│       ├── job_tracker.py
│       └── model_registry.py
│
├── tests\
│   ├── test_endpoints.py
│   └── test_integration.py
│
├── docs\
│   ├── tutorial_streamlit_fastapi.md
│   └── robust_tutorial.md
│
├── scripts\
│   ├── run_backend.ps1
│   ├── run_frontend.ps1
│   ├── run_mlflow.ps1
│   ├── run_tests.ps1
│   └── run_full_pipeline.ps1
│
├── requirements.txt
└── README-windows.md (this file)
```

---

## Environment Setup

### Step 1: Create Virtual Environment

```powershell
# Open PowerShell as Administrator (if you encounter permission issues)

# Navigate to your project directory
cd C:\Users\YourName\ml-fraud-dashboard

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

**Troubleshooting Execution Policy Error:**
```powershell
# If you get "cannot be loaded because running scripts is disabled"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then retry activation
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` prefix in your terminal.

### Step 2: Install Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install PyTorch CPU version (Windows-specific)
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install all other requirements
pip install -r requirements.txt

# Verify critical packages
python -c "import torch; import streamlit; import fastapi; print('✅ All imports successful')"
```

---

## Data Generation

### Generate Synthetic Transaction Data (~10k rows)

```powershell
# Activate virtual environment first
.\.venv\Scripts\Activate.ps1

# Generate synthetic data
python data\generate_synthetic_transactions.py

# Expected output:
# ✅ Generated 10,000 transactions
# ✅ Saved to data\transactions_raw.csv
# 📊 Fraud rate: 2.3%
```

**Verify data:**
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/transactions_raw.csv'); print(df.head(10))"
```

---

## Model Training

### Option 1: Train All Models (Full Pipeline)

```powershell
# Run the complete training pipeline
.\scripts\run_full_pipeline.ps1

# This will:
# 1. Generate features → feature_store\
# 2. Train sklearn Random Forest
# 3. Train PyTorch MLP with Skorch
# 4. Tune with Optuna (10 trials)
# 5. Save models to models\
# 6. Log experiments to MLflow
```

### Option 2: Train Individual Models

**Train Scikit-Learn Model:**
```powershell
python src\ml\train_sklearn.py --experiment-name "fraud_detection_sklearn"
```

**Train PyTorch + Skorch Model:**
```powershell
python src\ml\train_pytorch.py --epochs 50 --batch-size 128
```

**Hyperparameter Tuning with Optuna:**
```powershell
python src\ml\optuna_tune.py --n-trials 20 --timeout 600
```

---

## Running the Application

### Terminal 1: Start MLflow Tracking UI

```powershell
.\scripts\run_mlflow.ps1

# Or manually:
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# Access at: http://localhost:5000
```

### Terminal 2: Start FastAPI Backend

```powershell
# New PowerShell window
cd C:\Users\YourName\ml-fraud-dashboard
.\.venv\Scripts\Activate.ps1

.\scripts\run_backend.ps1

# Or manually:
uvicorn src.backend.app:app --host 0.0.0.0 --port 8000 --reload

# API Docs: http://localhost:8000/docs
# OpenAPI JSON: http://localhost:8000/openapi.json
```

### Terminal 3: Start Streamlit Frontend

```powershell
# New PowerShell window
cd C:\Users\YourName\ml-fraud-dashboard
.\.venv\Scripts\Activate.ps1

.\scripts\run_frontend.ps1

# Or manually:
streamlit run src\frontend\app.py --server.port 8501

# Dashboard: http://localhost:8501
```

### Access Points Summary

| Service | URL | Purpose |
|---------|-----|---------|
| **Streamlit Dashboard** | http://localhost:8501 | Main operator interface |
| **FastAPI Backend** | http://localhost:8000 | API endpoints |
| **FastAPI Docs** | http://localhost:8000/docs | Interactive API documentation |
| **MLflow UI** | http://localhost:5000 | Experiment tracking |

---

## Testing

### Run All Tests

```powershell
.\scripts\run_tests.ps1

# Or manually:
pytest tests\ -v --tb=short

# With coverage:
pytest tests\ --cov=src --cov-report=html
```

### Run Specific Test Files

```powershell
# Test FastAPI endpoints only
pytest tests\test_endpoints.py -v

# Test integration flows
pytest tests\test_integration.py -v
```

---

## MLOps Workflows

### View Experiment Tracking

```powershell
# MLflow UI should be running (Terminal 1)
# Navigate to http://localhost:5000

# Or query via CLI:
mlflow experiments list
mlflow runs list --experiment-id 0
```

### Feature Store Operations

```powershell
# Create new feature version
python -c "from src.ml.feature_engineering import create_features; create_features(version='v2')"

# List feature versions
dir feature_store\
```

### Model Registry

```powershell
# View registered models
python -c "from src.utils.model_registry import ModelRegistry; reg = ModelRegistry(); print(reg.list_models())"

# Promote model to production
python -c "from src.utils.model_registry import ModelRegistry; reg = ModelRegistry(); reg.promote_model('pytorch_mlp_v1', 'production')"
```

### Monitoring

```powershell
# View Prometheus metrics endpoint
curl http://localhost:8000/metrics

# Or in browser: http://localhost:8000/metrics
```

---

## Troubleshooting

### Issue: "ImportError: DLL load failed"

**Solution:**
```powershell
# Reinstall PyTorch CPU version
pip uninstall torch
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "Port already in use"

**Solution:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use different port:
uvicorn src.backend.app:app --port 8001
```

### Issue: "Streamlit gets stuck loading"

**Solution:**
```powershell
# Clear Streamlit cache
streamlit cache clear

# Restart with clean config
streamlit run src\frontend\app.py --server.headless true
```

### Issue: "ModuleNotFoundError"

**Solution:**
```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Issue: "MLflow database locked"

**Solution:**
```powershell
# Stop all MLflow processes
taskkill /IM mlflow.exe /F

# Remove lock file
Remove-Item mlruns\.mlflow.db-shm -ErrorAction SilentlyContinue
Remove-Item mlruns\.mlflow.db-wal -ErrorAction SilentlyContinue

# Restart MLflow
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

---

## Quick Start (All-in-One)

```powershell
# 1. Setup (one-time)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Generate data
python data\generate_synthetic_transactions.py

# 3. Train models
.\scripts\run_full_pipeline.ps1

# 4. Launch app (3 separate terminals)
# Terminal 1:
.\scripts\run_mlflow.ps1

# Terminal 2:
.\scripts\run_backend.ps1

# Terminal 3:
.\scripts\run_frontend.ps1

# 5. Access dashboard
# Open browser: http://localhost:8501
```

---

## Next Steps

1. **Read Tutorial**: See `docs\tutorial_streamlit_fastapi.md` for step-by-step guide
2. **Production Hardening**: See `docs\robust_tutorial.md` for MLOps best practices
3. **Customize Models**: Modify hyperparameters in `src\ml\optuna_tune.py`
4. **Add Features**: Extend feature engineering in `src\ml\feature_engineering.py`
5. **Deploy**: Consider Docker containerization (see `docs\robust_tutorial.md`)

---

## Support & Resources

- **Documentation**: `docs\` folder
- **API Reference**: http://localhost:8000/docs (when backend is running)
- **MLflow Experiments**: http://localhost:5000 (when MLflow UI is running)

**Happy Building! 🚀**
