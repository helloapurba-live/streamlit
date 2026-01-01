# 🧠 Mental Model & Mind Map: Production ML Dashboard

## 🎯 Core Concept Map

```
                    PRODUCTION ML DASHBOARD
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    FRONTEND            BACKEND              ML/MLOps
        │                   │                   │
   ┌────┴────┐         ┌────┴────┐        ┌────┴────┐
   │         │         │         │        │         │
Streamlit  UX      FastAPI  Jobs    Models    Tracking
   │         │         │         │        │         │
Multi-   State   Pydantic Async  PyTorch  MLflow
Page   Management Validation Tasks  Skorch  Feature
                                    Optuna   Store
                                    SHAP     Registry
```

---

## 📋 Quick Recall Summary

### **The Problem**
Real-time fraud detection requiring:
- <50ms single predictions
- Batch processing (10K+ rows)
- Explainability (SHAP)
- Continuous improvement (hyperparameter tuning)
- Operator-friendly UX (dashboard)

### **The Stack** (Why Each Tool)

| Tool | Purpose | Key Benefit |
|------|---------|-------------|
| **Streamlit** | Frontend dashboard | Zero-config, 30 lines → full UI |
| **FastAPI** | REST API backend | Auto-docs, async, Pydantic validation |
| **PyTorch** | Deep learning | Flexibility, production-ready |
| **Skorch** | PyTorch → sklearn | Pipeline compatibility, GridSearch, Optuna |
| **Optuna** | Hyperparameter tuning | Bayesian optimization (10 trials vs 100+) |
| **MLflow** | Experiment tracking | Git for models, versioning |
| **SHAP** | Explainability | Feature importance, compliance |
| **Prometheus** | Monitoring | Metrics > logs in production |

### **The Architecture** (5 Layers)

```
┌─────────────────────────────────────────────────────────┐
│ LAYER 1: PRESENTATION (Streamlit)                       │
│ - Multi-page navigation (5 pages)                       │
│ - Real-time polling for batch jobs                      │
│ - Interactive charts (Plotly)                           │
└─────────────────────────────────────────────────────────┘
                         │ HTTP
┌─────────────────────────────────────────────────────────┐
│ LAYER 2: API (FastAPI)                                  │
│ - POST /predict_single → Real-time scoring              │
│ - POST /predict_batch → Background jobs                 │
│ - GET /job_status/{id} → Job polling                    │
│ - Pydantic validation, Prometheus metrics               │
└─────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────┐
│ LAYER 3: BUSINESS LOGIC                                 │
│ - Feature extraction                                    │
│ - Model inference (cached)                              │
│ - SHAP explanation (cached)                             │
│ - Job orchestration (SQLite/Redis)                      │
└─────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────┐
│ LAYER 4: ML MODELS                                      │
│ - sklearn RandomForest (baseline)                       │
│ - PyTorch MLP via Skorch (deep learning)                │
│ - Pipeline: StandardScaler → Model                      │
│ - Trained with SMOTE (imbalanced data)                  │
└─────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────┐
│ LAYER 5: DATA & MLOPS                                   │
│ - Feature store (Parquet, versioned)                    │
│ - Model registry (joblib + metadata)                    │
│ - Experiment tracking (MLflow)                          │
│ - Training data (synthetic generator)                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Concepts Mind Map

### **1. Skorch: PyTorch ↔ sklearn Bridge**

```
PyTorch Model (nn.Module)
        │
        │ Wrap with Skorch
        ▼
NeuralNetClassifier
        │
        ├─ .fit(X, y)           ✅ sklearn compatible
        ├─ .predict(X)          ✅ sklearn compatible
        ├─ .predict_proba(X)    ✅ sklearn compatible
        │
        ├─ Works in Pipeline    ✅ StandardScaler → Model
        ├─ Works in GridSearch  ✅ Hyperparameter tuning
        └─ Works with Optuna    ✅ Bayesian optimization
```

**Mental Model**: Skorch is an adapter pattern. PyTorch speaks "tensors and gradients," sklearn speaks "fit/predict." Skorch translates.

**Critical Bug to Avoid**:
```python
# ❌ WRONG: Don't apply softmax when using CrossEntropyLoss
def forward(self, x):
    return F.softmax(self.fc(x), dim=-1)

# ✅ CORRECT: Return raw logits
def forward(self, x):
    return self.fc(x)  # CrossEntropyLoss applies log-softmax internally
```

---

### **2. FastAPI: Request → Validation → Response**

```
HTTP Request
    │
    ├─ Pydantic Model validates
    │  (type checking, range validation, auto-docs)
    │
    ├─ Endpoint function executes
    │  (business logic, model inference)
    │
    ├─ Prometheus records metrics
    │  (latency, counts, errors)
    │
    └─ JSON Response
       (with proper HTTP status codes)
```

**Mental Model**: Pydantic is a type-safe wall. Invalid requests never reach your code.

**Pattern**:
```python
class Transaction(BaseModel):
    amount: float = Field(gt=0)  # Must be positive

@app.post("/predict")
def predict(txn: Transaction):  # Already validated!
    return model.predict(...)
```

---

### **3. Streamlit: Script Reruns on Interaction**

```
User loads page
    │
    ├─ Script runs top-to-bottom
    │
User clicks button
    │
    ├─ Entire script reruns
    │  (not just button handler)
    │
Use st.session_state to persist data
    │
    └─ Values survive across reruns
```

**Mental Model**: Streamlit is like a React component that rerenders on every state change, but for Python.

**Pattern**:
```python
# Save state across reruns
if 'job_id' not in st.session_state:
    st.session_state.job_id = None

if st.button("Submit"):
    st.session_state.job_id = submit_job()
    st.rerun()  # Force immediate rerun
```

---

### **4. Optuna: Bayesian Optimization**

```
Trial 1: lr=0.01, hidden=64  → ROC-AUC=0.85
    │
    ├─ Optuna learns: "Lower lr might be better"
    │
Trial 2: lr=0.001, hidden=128 → ROC-AUC=0.90
    │
    ├─ Optuna learns: "lr=0.001 good, try bigger network"
    │
Trial 3: lr=0.001, hidden=256 → ROC-AUC=0.92
    │
    └─ Converges to optimum in 10 trials (vs 100+ for grid search)
```

**Mental Model**: Optuna is GPS navigation. GridSearch is randomly driving.

**Pattern**:
```python
def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)  # Log scale!
    model = NeuralNetClassifier(..., lr=lr)
    return cross_val_score(model, X, y, cv=3).mean()

study.optimize(objective, n_trials=10)
```

---

### **5. MLflow: Experiment Tracking**

```
Experiment: "fraud_detection_pytorch"
    │
    ├─ Run 1 (2025-01-20 10:00)
    │   ├─ Params: lr=0.001, batch_size=128
    │   ├─ Metrics: roc_auc=0.89, precision=0.87
    │   └─ Artifacts: model.pkl, plots/
    │
    ├─ Run 2 (2025-01-20 14:30)
    │   ├─ Params: lr=0.0005, batch_size=256
    │   ├─ Metrics: roc_auc=0.91, precision=0.89
    │   └─ Artifacts: model.pkl, plots/
    │
    └─ Compare runs, download best model
```

**Mental Model**: MLflow is Git for models. Commit = Run.

**Pattern**:
```python
with mlflow.start_run():
    mlflow.log_params({"lr": 0.001, "epochs": 50})
    model.fit(X, y)
    mlflow.log_metric("roc_auc", 0.91)
    mlflow.sklearn.log_model(model, "model")
```

---

### **6. SHAP: Model Explanation**

```
Prediction: 78% fraud probability
    │
Why?
    │
    ├─ amount (+0.25)         → 7x customer baseline
    ├─ is_night (+0.12)       → Transaction at 11 PM
    ├─ distance (+0.08)       → 150 miles from home
    └─ merchant_risk (+0.05)  → High-risk merchant category
```

**Mental Model**: SHAP values = "How much does changing this feature change the prediction?"

**Pattern**:
```python
import shap

explainer = shap.TreeExplainer(model)  # For tree models
shap_values = explainer.shap_values(X_test)

# For single prediction
shap_values[1][0]  # Fraud class, first sample
# → [0.25, 0.12, 0.08, ...]  # Feature contributions
```

---

## 🎓 Production Patterns

### **Pattern 1: Model Caching**

```python
# ❌ BAD: Load on every request (500ms)
@app.post("/predict")
def predict(txn: Transaction):
    model = joblib.load("model.pkl")  # Disk I/O every time!
    return model.predict(...)

# ✅ GOOD: Load once at startup (5ms)
model = joblib.load("model.pkl")  # Load once globally

@app.post("/predict")
def predict(txn: Transaction):
    return model.predict(...)  # Memory access
```

### **Pattern 2: Background Jobs**

```python
# ❌ BAD: Block API request
@app.post("/predict_batch")
def predict_batch(file: UploadFile):
    df = pd.read_csv(file.file)  # Blocks for 30 seconds!
    predictions = model.predict(df)
    return predictions

# ✅ GOOD: Return immediately, process in background
@app.post("/predict_batch")
async def predict_batch(file: UploadFile, bg: BackgroundTasks):
    job_id = uuid.uuid4()
    bg.add_task(process_batch, job_id, file)
    return {"job_id": job_id, "status": "pending"}
```

### **Pattern 3: Training-Serving Skew**

```python
# Problem: Model trained on customer_avg_amount feature
# But how do we compute this at serving time?

# ❌ BAD: Use dummy value
features = [txn.amount, 100.0, ...]  # Wrong distribution!

# ✅ GOOD: Maintain feature cache
feature_cache = {
    "CUST_001": {"avg_amount": 120.0, "std": 45.0},
    # ... updated daily from transaction history
}

features = [
    txn.amount,
    feature_cache[txn.customer_id]["avg_amount"],
    ...
]
```

---

## 🚨 Common Pitfalls

### **Pitfall 1: PyTorch + CrossEntropyLoss**
❌ Applying softmax in forward() → destroys gradients
✅ Return logits, let CrossEntropyLoss handle softmax

### **Pitfall 2: Streamlit State**
❌ Using global variables → lost on rerun
✅ Using st.session_state → persists

### **Pitfall 3: Optuna Search Space**
❌ Linear search for learning rate (0.001, 0.002, ...)
✅ Log scale search (1e-4, 1e-3, 1e-2) using `log=True`

### **Pitfall 4: FastAPI Blocking**
❌ Long operations in endpoint → API freezes
✅ BackgroundTasks for async processing

### **Pitfall 5: Model Versioning**
❌ Overwriting model.pkl → no rollback
✅ Version models (model_v1.pkl, model_v2.pkl) + metadata

---

## 📊 Decision Trees

### **When to Use What?**

```
Need to train model?
    │
    ├─ Tabular data + interpretability → sklearn RandomForest
    ├─ Tabular data + performance → XGBoost/LightGBM
    ├─ Deep learning required → PyTorch + Skorch
    └─ Time series → LSTM/Transformer (PyTorch)

Need to tune hyperparameters?
    │
    ├─ <5 parameters, discrete → GridSearch
    ├─ >5 parameters, continuous → Optuna
    └─ Neural networks → Optuna (Bayesian >> Grid)

Need to deploy model?
    │
    ├─ Simple API → FastAPI
    ├─ Dashboard → Streamlit
    ├─ High throughput → Triton Inference Server
    └─ Edge devices → ONNX Runtime

Need to explain predictions?
    │
    ├─ Tree models → SHAP TreeExplainer (fast)
    ├─ Linear models → Coefficients
    ├─ Deep learning → SHAP KernelExplainer (slow)
    └─ Any model → LIME (model-agnostic)
```

---

## 🎯 Mental Shortcuts

### **The 90/10 Rule**
- 10% of effort: Build model with 90% accuracy
- 90% of effort: Deploy at <50ms latency, monitor, retrain, explain

### **The CAP Theorem of ML**
You can optimize for 2 of 3:
- **Speed**: Fast inference (<50ms)
- **Accuracy**: High performance (>95% ROC-AUC)
- **Interpretability**: Explainable decisions (SHAP)

Choose: Fast + Accurate = Deep learning (black box)
       Fast + Interpretable = Logistic regression
       Accurate + Interpretable = RandomForest with SHAP

### **The Validation Hierarchy**
1. **Type validation**: Pydantic (compile-time)
2. **Business logic validation**: Custom validators (runtime)
3. **Model validation**: Cross-validation (training-time)
4. **Production validation**: A/B testing (deploy-time)

---

## 🔄 Workflow Memory Aids

### **Training Workflow**
```
Data → Features → Split → SMOTE → Train → Validate → Log → Save
 ↓       ↓         ↓       ↓       ↓       ↓        ↓     ↓
CSV   Engineer  80/20  Balance  Model    CV     MLflow  .pkl
```

### **Inference Workflow**
```
Request → Validate → Extract → Scale → Predict → Explain → Log → Response
   ↓        ↓          ↓        ↓        ↓         ↓       ↓       ↓
 JSON    Pydantic   Features  Same    Model     SHAP  Prometheus JSON
                             as train
```

### **Deployment Workflow**
```
Code → Test → Docker → Deploy → Monitor → Alert → Rollback
 ↓      ↓       ↓        ↓        ↓        ↓        ↓
Git   pytest  Build   k8s/VM  Prometheus Slack  Previous
                                                  version
```

---

## 📝 Cheat Sheet Formulas

### **Model Performance**
- **Precision** = TP / (TP + FP) → "Of predicted fraud, how many were actual fraud?"
- **Recall** = TP / (TP + FN) → "Of actual fraud, how many did we catch?"
- **F1** = 2 × (Precision × Recall) / (Precision + Recall)
- **ROC-AUC** = Area under curve → "Probability model ranks fraud > normal"

### **API Performance**
- **Latency (p99)** = 99th percentile response time
- **Throughput** = Requests per second (RPS)
- **Error Rate** = Failed requests / Total requests
- **Availability** = Uptime / (Uptime + Downtime)

### **Cost-Benefit**
- **False Positive Cost** = Legitimate transaction blocked → customer frustration
- **False Negative Cost** = Fraud not caught → financial loss
- **Threshold** = Optimize based on cost ratio

---

## 🧩 Integration Map

```
              ┌─────────────┐
              │  Streamlit  │
              └──────┬──────┘
                     │ HTTP (requests)
              ┌──────▼──────┐
              │   FastAPI   │
              └──────┬──────┘
                     │ joblib.load()
          ┌──────────┼──────────┐
          ▼                     ▼
    ┌──────────┐         ┌──────────┐
    │  sklearn │         │  PyTorch │
    │   Model  │         │ + Skorch │
    └──────────┘         └──────────┘
          │                     │
          └──────────┬──────────┘
                     │ MLflow
              ┌──────▼──────┐
              │  Experiment │
              │   Tracking  │
              └─────────────┘
```

---

## 🎬 The Big Picture

**You're building a real-time decision system that:**
1. Scores transactions in <50ms (FastAPI + cached models)
2. Explains decisions to compliance (SHAP)
3. Processes batches asynchronously (BackgroundTasks)
4. Improves continuously (Optuna + MLflow)
5. Serves operators visually (Streamlit)

**Key insight**: This isn't a "model in a Jupyter notebook." It's a **full-stack ML application** with frontend, backend, ML, MLOps, and monitoring.

---

## 🚀 Quick Reference Commands

```bash
# Data generation
python data/generate_synthetic_transactions.py

# Training
python src/ml/train_sklearn.py      # RandomForest
python src/ml/train_pytorch.py      # PyTorch + Skorch
python src/ml/optuna_tune.py        # Hyperparameter tuning

# Deployment
uvicorn src.backend.app:app --reload         # Backend
streamlit run src/frontend/app.py            # Frontend
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db  # MLflow

# Testing
pytest tests/ -v --cov=src
```

---

## 💡 Interview Talking Points

**When asked "How do you deploy ML models?"**
> "I build FastAPI endpoints with Pydantic validation for type safety. Models are loaded once at startup for <50ms latency. For batch processing, I use BackgroundTasks to avoid blocking the API. I monitor with Prometheus metrics (latency, throughput, error rate) and explain predictions with SHAP for compliance."

**When asked "How do you tune hyperparameters?"**
> "For deep learning, I use Optuna's Bayesian optimization instead of GridSearch—it converges in 10 trials vs 100+. Skorch makes PyTorch models sklearn-compatible, so they work seamlessly with Optuna. I track all experiments in MLflow to compare runs and reproduce results."

**When asked "How do you build ML dashboards?"**
> "I use Streamlit for operator UIs—it's 30 lines of Python vs 300 lines of React. The dashboard polls FastAPI endpoints for batch job status, displays SHAP explanations visually, and uses session_state to persist data across reruns. This gives operators a production-ready interface without frontend engineering."

---

**Remember**: Production ML is 10% model building, 90% engineering. Master the 90%.
