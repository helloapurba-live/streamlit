# 🚀 Quick Start Guide - Bank Transaction Fraud Detection Dashboard

## ✅ What Has Been Built

A **production-ready ML web application** for bank transaction fraud detection with:

### 🎨 Frontend (Streamlit Dashboard)
- **5 Interactive Pages**:
  1. 📊 **Overview**: Real-time metrics, fraud rate gauge, alert dashboard
  2. 🔍 **Single Prediction**: Manual transaction input with SHAP explanations
  3. 📦 **Batch Processing**: CSV upload, job tracking, result download
  4. 🕵️ **Investigation**: Customer deep-dive with AI-powered insights
  5. ⚙️ **Admin**: System health, MLflow links, model management

### 🔧 Backend (FastAPI REST API)
- **6 Core Endpoints**:
  - `POST /predict_single` - Real-time fraud prediction
  - `POST /predict_batch` - Async batch processing
  - `GET /job_status/{job_id}` - Poll job progress
  - `GET /download_result/{job_id}` - Download results
  - `POST /agent_investigate` - LLM-powered investigation
  - `GET /metrics` - Prometheus monitoring
- **Features**: Pydantic validation, background tasks, SQLite job tracking

### 🤖 Machine Learning Stack
- **Scikit-Learn**: Random Forest with SMOTE for imbalanced data
- **PyTorch + Skorch**: MLP wrapped for sklearn compatibility
- **Optuna**: Bayesian hyperparameter optimization
- **SHAP**: Model explainability and feature importance

### 🔬 MLOps Components
- **MLflow**: Experiment tracking and model versioning
- **Feature Store**: Parquet-based versioned feature storage
- **Model Registry**: Local model management with metadata
- **Monitoring**: Prometheus metrics integration

---

## 🏃 Quick Start (5 Minutes)

### Step 1: Setup Environment

```powershell
# Clone/navigate to project
cd C:\path\to\streamlit

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Generate Data & Train Models

```powershell
# Run complete pipeline (generates data + trains all models)
.\scripts\run_full_pipeline.ps1
```

**Expected output**:
```
Generating 10,000 transactions...
✅ Generated 10,000 transactions
✅ Saved to: data\transactions_raw.csv

Training sklearn Random Forest...
🎯 ROC-AUC: 0.92
✅ Model saved to models\sklearn_rf_model.pkl

Training PyTorch + Skorch MLP...
🎯 ROC-AUC: 0.89
✅ Model saved to models\pytorch_mlp_model.pkl

Hyperparameter tuning with Optuna...
🎯 Best ROC-AUC: 0.91
✅ Tuning complete!
```

### Step 3: Launch Application (3 Terminals)

**Terminal 1: MLflow UI**
```powershell
.\scripts\run_mlflow.ps1
# Access: http://localhost:5000
```

**Terminal 2: FastAPI Backend**
```powershell
.\scripts\run_backend.ps1
# Access: http://localhost:8000/docs
```

**Terminal 3: Streamlit Dashboard**
```powershell
.\scripts\run_frontend.ps1
# Access: http://localhost:8501
```

### Step 4: Use the Dashboard

1. **Open browser**: http://localhost:8501
2. **Overview page**: See metrics and fraud rate gauge
3. **Single Prediction**: Try predicting a transaction
4. **Batch Processing**: Upload a CSV file (use `data/transactions_raw.csv` as template)

---

## 📁 Project Structure

```
streamlit/
├── data/
│   └── generate_synthetic_transactions.py    # Data generator
├── src/
│   ├── backend/
│   │   └── app.py                            # FastAPI application
│   ├── frontend/
│   │   └── app.py                            # Streamlit dashboard
│   └── ml/
│       ├── train_sklearn.py                  # Random Forest training
│       ├── train_pytorch.py                  # PyTorch + Skorch training
│       └── optuna_tune.py                    # Hyperparameter tuning
├── scripts/
│   ├── run_full_pipeline.ps1                 # Complete training pipeline
│   ├── run_backend.ps1                       # Start FastAPI
│   ├── run_frontend.ps1                      # Start Streamlit
│   ├── run_mlflow.ps1                        # Start MLflow UI
│   └── run_tests.ps1                         # Run pytest suite
├── tests/
│   └── test_endpoints.py                     # API tests
├── docs/
│   └── tutorial_streamlit_fastapi.md         # Complete tutorial
├── requirements.txt                          # Python dependencies
└── README-windows.md                         # Detailed setup guide
```

---

## 🎯 Key Features Demonstrated

### 1. Dashboard-First UX
- Multi-page Streamlit application with session state
- Real-time polling for batch job progress
- Interactive Plotly charts and gauges
- Color-coded alerts (green/yellow/red)

### 2. ML Model Integration
- **Skorch**: PyTorch models with sklearn API
- **Optuna**: Efficient hyperparameter search
- **SHAP**: Explainability visualizations
- **Imbalanced-learn**: SMOTE for class imbalance

### 3. Production Patterns
- **Validation**: Pydantic models for type safety
- **Async Processing**: Background tasks for batch jobs
- **Job Tracking**: SQLite database for state persistence
- **Monitoring**: Prometheus metrics endpoints
- **Error Handling**: Graceful failures with user-friendly messages

### 4. MLOps Best Practices
- **Reproducibility**: Pinned dependencies, random seeds
- **Tracking**: MLflow logs params, metrics, artifacts
- **Versioning**: Feature store with timestamps
- **Testing**: Pytest suite for API endpoints

---

## 📊 Sample Use Cases

### Use Case 1: Real-Time Transaction Scoring

**Scenario**: A $850 online purchase at 11 PM on Saturday

**Steps**:
1. Navigate to "🔍 Single Prediction" page
2. Enter transaction details:
   - Amount: $850
   - Category: online_shopping
   - Hour: 23, Weekend: Yes
3. Click "Analyze Transaction"

**Result**:
```
🚨 FRAUD ALERT
Fraud Probability: 78.5%
Recommended Action: BLOCK or REVIEW

Top Contributing Features:
  • amount: +0.25 (high value)
  • is_night: +0.12 (unusual time)
  • is_online: +0.08 (online risk)
```

### Use Case 2: Batch Review of Daily Transactions

**Scenario**: Review all transactions from the last 24 hours

**Steps**:
1. Navigate to "📦 Batch Processing" page
2. Upload CSV with transaction records
3. Wait for job completion (progress bar updates)
4. Download results with fraud probabilities
5. Filter for high-risk transactions (probability > 70%)

**Result**: CSV file with fraud predictions for each transaction

### Use Case 3: Customer Investigation

**Scenario**: Deep-dive into flagged customer account

**Steps**:
1. Navigate to "🕵️ Investigation" page
2. Enter Customer ID (e.g., CUST_000123)
3. Review transaction history
4. Run AI investigation with query: "Why was this account flagged?"

**Result**:
```
Risk Score: 75% (HIGH RISK)

Key Findings:
  • 3 high-value transactions ($800-$1200) in 24 hours
  • Transactions in multiple states (CA, NY)
  • New merchant categories not seen in 90-day history
  • Velocity anomaly: 5 transactions in 2-hour window

Recommended Actions:
  • Contact customer for verification
  • Place temporary hold on card
  • Review with fraud team
```

---

## 🔧 Customization Guide

### Add New Merchant Categories

**File**: `data/generate_synthetic_transactions.py`

```python
self.merchant_categories = [
    'grocery', 'gas_transport', 'restaurant', 'entertainment',
    'online_shopping', 'bills_utilities', 'health_fitness', 'travel',
    'electronics',  # ← Add new category
    'clothing'      # ← Add new category
]
```

### Tune Model Hyperparameters

**File**: `src/ml/optuna_tune.py`

```python
# Modify search space
hidden_dim1 = trial.suggest_int('hidden_dim1', 32, 256)  # ← Increase range
dropout = trial.suggest_float('dropout', 0.1, 0.7)       # ← Adjust range
```

### Add New Dashboard Page

**File**: `src/frontend/app.py`

```python
def page_analytics():
    st.markdown("<h1>📈 Analytics Dashboard</h1>")
    # Add your custom visualizations here
    ...

# In main() function:
page = st.sidebar.radio(
    "Select Page",
    options=["📊 Overview", "🔍 Single Prediction", "📈 Analytics", ...]
)

if page == "📈 Analytics":
    page_analytics()
```

---

## 🧪 Testing

```powershell
# Run all tests
.\scripts\run_tests.ps1

# Run specific test file
pytest tests\test_endpoints.py -v

# Run with coverage
pytest tests\ --cov=src --cov-report=html
```

**Test Coverage**:
- ✅ API endpoint validation
- ✅ Pydantic model validation
- ✅ Health check endpoints
- ✅ Single prediction flow
- ✅ Investigation endpoint

---

## 📚 Documentation

### Comprehensive Guides

1. **README-windows.md**
   - Detailed Windows 11 setup instructions
   - Troubleshooting common issues
   - PowerShell command reference

2. **docs/tutorial_streamlit_fastapi.md**
   - Step-by-step tutorial with theory
   - Architecture diagrams
   - Code explanations
   - Sample inputs/outputs
   - Production considerations

3. **Code Comments**
   - Every file has teaching-style docstrings
   - Line-by-line explanations
   - Learning objectives and theory sections

---

## 🚨 Troubleshooting

### Issue: Backend won't start

```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process if needed
taskkill /PID <PID> /F

# Try different port
uvicorn src.backend.app:app --port 8001
```

### Issue: Models not found

```powershell
# Check if models directory exists
ls models\

# If empty, run training pipeline
.\scripts\run_full_pipeline.ps1
```

### Issue: MLflow database locked

```powershell
# Stop MLflow
taskkill /IM mlflow.exe /F

# Remove lock files
Remove-Item mlruns\.mlflow.db-shm
Remove-Item mlruns\.mlflow.db-wal

# Restart MLflow
.\scripts\run_mlflow.ps1
```

---

## 🎓 Learning Path

### Beginner Level
1. Run the full pipeline and explore the dashboard
2. Read README-windows.md for setup details
3. Try single prediction page with different values
4. Review data generator code to understand features

### Intermediate Level
1. Read tutorial_streamlit_fastapi.md for architecture
2. Modify Optuna search space and retrain
3. Add new merchant categories to data generator
4. Customize Streamlit dashboard styling

### Advanced Level
1. Implement custom fraud patterns in generator
2. Add new ML model (e.g., XGBoost, LightGBM)
3. Create custom feature engineering functions
4. Add model performance monitoring dashboard
5. Implement A/B testing for model versions

---

## 🚀 Next Steps

### Immediate (Next Hour)
- [ ] Explore all 5 dashboard pages
- [ ] Try batch processing with sample CSV
- [ ] View MLflow experiments
- [ ] Review API docs at http://localhost:8000/docs

### Short-Term (Next Day)
- [ ] Read complete tutorial documentation
- [ ] Modify hyperparameters and retrain
- [ ] Add custom fraud patterns to data generator
- [ ] Run test suite and understand coverage

### Medium-Term (Next Week)
- [ ] Implement additional ML models
- [ ] Add data quality checks with Great Expectations
- [ ] Create model retraining pipeline with Prefect
- [ ] Add authentication to API endpoints

### Long-Term (Production)
- [ ] Containerize with Docker
- [ ] Set up CI/CD pipeline
- [ ] Deploy to cloud (Azure/AWS/GCP)
- [ ] Implement full monitoring stack (Prometheus + Grafana)
- [ ] Add A/B testing framework

---

## 💡 Tips & Best Practices

### Performance
- **Batch Size**: For large CSVs, increase batch_size in backend to 500-1000
- **Model Loading**: Models load on startup; restart backend after retraining
- **Caching**: Streamlit caches API responses; clear with `st.cache_clear()`

### Development
- **Hot Reload**: Backend uses `--reload` flag; changes auto-reload
- **Debugging**: Check backend terminal for error logs
- **Testing**: Run tests before committing changes

### MLOps
- **Versioning**: Always version your feature datasets
- **Logging**: Log all hyperparameters to MLflow
- **Monitoring**: Check Prometheus metrics regularly

---

## 📞 Support & Resources

### Code Repository
- Branch: `claude/ml-dashboard-app-01Mea12CobvhmoQU4N5w3689`
- Commit: `03121cb`

### Documentation
- **In-Code**: Every file has comprehensive docstrings
- **README**: README-windows.md for setup
- **Tutorial**: docs/tutorial_streamlit_fastapi.md for deep-dive

### External Resources
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Streamlit Gallery**: https://streamlit.io/gallery
- **Skorch Docs**: https://skorch.readthedocs.io
- **Optuna Tutorials**: https://optuna.readthedocs.io

---

## ✅ Checklist: Verify Everything Works

```powershell
# 1. Environment setup
python --version  # Should be 3.10 or 3.11
pip list | findstr "streamlit fastapi torch skorch optuna"

# 2. Data generation
python data\generate_synthetic_transactions.py
ls data\transactions_raw.csv  # Should exist

# 3. Model training
.\scripts\run_full_pipeline.ps1
ls models\*.pkl  # Should see sklearn_rf_model.pkl and pytorch_mlp_model.pkl

# 4. Launch services
.\scripts\run_mlflow.ps1      # Terminal 1
.\scripts\run_backend.ps1     # Terminal 2
.\scripts\run_frontend.ps1    # Terminal 3

# 5. Verify endpoints
curl http://localhost:5000     # MLflow
curl http://localhost:8000/health  # Backend
curl http://localhost:8501     # Frontend (or open in browser)

# 6. Run tests
.\scripts\run_tests.ps1
```

---

## 🎉 Congratulations!

You now have a **complete, production-ready ML web application** running locally on Windows 11!

**What you've built**:
✅ Dashboard-first fraud detection system
✅ Real-time and batch prediction capabilities
✅ Deep learning models with sklearn compatibility
✅ Hyperparameter optimization pipeline
✅ Experiment tracking and model management
✅ Model explainability with SHAP
✅ Production patterns (validation, monitoring, testing)

**Happy Building! 🚀**
