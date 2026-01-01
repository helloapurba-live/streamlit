# 📋 ML Dashboard Interview Cheat Sheet

## 🎯 Quick Reference for Technical Interviews

---

## 1️⃣ STACK OVERVIEW

| Component | Technology | Why This Choice | Alternative |
|-----------|-----------|-----------------|-------------|
| **Frontend** | Streamlit | Zero-config, 30 lines → full UI | React (300+ lines) |
| **Backend** | FastAPI | Auto-docs, async, Pydantic | Flask (manual validation) |
| **DL Framework** | PyTorch | Flexibility, research → production | TensorFlow |
| **sklearn Bridge** | Skorch | PyTorch → sklearn API | Manual wrapper |
| **Tuning** | Optuna | Bayesian (10 trials vs 100+) | GridSearch (exhaustive) |
| **Tracking** | MLflow | Git for models, versioning | Weights & Biases |
| **Explainability** | SHAP | Model-agnostic, values | LIME (slower) |
| **Monitoring** | Prometheus | Industry standard, metrics | Datadog (paid) |
| **Imbalance** | SMOTE | Synthetic minority oversampling | Class weights |

---

## 2️⃣ CODE SNIPPETS

### **Skorch: PyTorch → sklearn**

```python
from skorch import NeuralNetClassifier
import torch.nn as nn

# 1. Define PyTorch model
class FraudMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)  # Binary classification
        )

    def forward(self, x):
        return self.layers(x)  # Return LOGITS (not softmax!)

# 2. Wrap with Skorch
net = NeuralNetClassifier(
    FraudMLP,
    criterion=nn.CrossEntropyLoss,  # Expects logits
    optimizer=torch.optim.Adam,
    lr=0.001,
    max_epochs=50,
    batch_size=128
)

# 3. Use like sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('net', net)
])

pipeline.fit(X_train, y_train)  # ✅ Works!
y_proba = pipeline.predict_proba(X_test)  # ✅ Works!
```

**Interview Tip**: Emphasize that `forward()` returns logits, not softmax. CrossEntropyLoss applies log-softmax internally.

---

### **FastAPI: Request Validation**

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

# Pydantic model (auto-validates!)
class Transaction(BaseModel):
    amount: float = Field(gt=0, example=100.0)  # Must be positive
    merchant_category: str = Field(example="grocery")
    hour: int = Field(ge=0, le=23)  # 0-23 range

@app.post("/predict")
def predict(txn: Transaction):  # Already validated!
    # txn.amount is guaranteed to be float > 0
    features = [txn.amount, txn.hour, ...]
    return {"fraud_probability": model.predict([features])[0]}
```

**Interview Tip**: "Pydantic provides compile-time type safety. Invalid requests return 422 before reaching my code."

---

### **Optuna: Bayesian Hyperparameter Tuning**

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)  # Log scale!
    hidden = trial.suggest_int('hidden_dim', 32, 128)

    # Build model
    net = NeuralNetClassifier(
        FraudMLP,
        module__hidden_dim=hidden,
        lr=lr,
        max_epochs=20
    )

    # Cross-validate
    scores = cross_val_score(net, X_train, y_train, cv=3, scoring='roc_auc')
    return scores.mean()

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print(f"Best ROC-AUC: {study.best_value:.3f}")
print(f"Best params: {study.best_params}")
```

**Interview Tip**: "Optuna uses Tree-structured Parzen Estimator (TPE) to learn which hyperparameters work well. It converges 10x faster than GridSearch."

---

### **Streamlit: Multi-Page with State**

```python
import streamlit as st

st.set_page_config(page_title="Fraud Detection", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Select Page", ["Overview", "Predict", "Batch"])

# Persist state across reruns
if 'job_id' not in st.session_state:
    st.session_state.job_id = None

if page == "Predict":
    with st.form("predict_form"):
        amount = st.number_input("Amount", value=100.0)
        submitted = st.form_submit_button("Analyze")

    if submitted:
        # Call API
        result = requests.post("http://localhost:8000/predict", json={
            "amount": amount, ...
        }).json()

        if result["fraud_probability"] > 0.5:
            st.error(f"🚨 FRAUD ({result['fraud_probability']*100:.1f}%)")
        else:
            st.success(f"✅ LEGITIMATE")
```

**Interview Tip**: "Streamlit reruns the entire script on each interaction. I use `st.session_state` to persist data and `st.form` to batch inputs."

---

### **MLflow: Experiment Tracking**

```python
import mlflow

mlflow.set_experiment("fraud_detection")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 50
    })

    # Train model
    model.fit(X_train, y_train)

    # Log metrics
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    mlflow.log_metric("roc_auc", roc_auc)

    # Save model
    mlflow.sklearn.log_model(model, "model")

# View in UI: mlflow ui --port 5000
```

**Interview Tip**: "MLflow is Git for models. Each run is a commit. I can compare experiments, download models, and promote to production."

---

### **SHAP: Model Explanation**

```python
import shap

# For tree models (fast)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For fraud class (binary classification)
fraud_shap = shap_values[1]  # Index 1 = fraud class

# Single prediction explanation
sample_idx = 0
feature_names = ['amount', 'hour', 'distance', ...]

# Top 3 contributors
contributions = list(zip(feature_names, fraud_shap[sample_idx]))
contributions.sort(key=lambda x: abs(x[1]), reverse=True)

print("Top 3 reasons for fraud prediction:")
for feature, shap_value in contributions[:3]:
    print(f"  {feature}: {shap_value:+.3f}")

# Visualization
shap.summary_plot(fraud_shap, X_test, feature_names=feature_names)
```

**Interview Tip**: "SHAP values show how much each feature contributes to the prediction. Positive = increases fraud probability. Negative = decreases."

---

### **Background Jobs: Async Batch Processing**

```python
from fastapi import BackgroundTasks
import uuid

jobs = {}  # In production: Redis or PostgreSQL

@app.post("/predict_batch")
async def predict_batch(file: UploadFile, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0}

    # Schedule background processing
    background_tasks.add_task(process_batch, job_id, file)

    return {"job_id": job_id}

def process_batch(job_id: str, file: UploadFile):
    df = pd.read_csv(file.file)
    total = len(df)

    for i, chunk in enumerate(np.array_split(df, 10)):
        predictions = model.predict_proba(chunk)[:, 1]
        chunk['fraud_probability'] = predictions

        # Update progress
        jobs[job_id]["progress"] = int((i + 1) / 10 * 100)

    jobs[job_id]["status"] = "completed"

@app.get("/job_status/{job_id}")
def get_status(job_id: str):
    return jobs[job_id]
```

**Interview Tip**: "BackgroundTasks prevents blocking the API. The endpoint returns immediately with a job_id. Client polls /job_status for progress."

---

## 3️⃣ ARCHITECTURE PATTERNS

### **Pattern 1: Model Serving**

```python
# ❌ BAD: Load model on every request (500ms latency)
@app.post("/predict")
def predict(txn: Transaction):
    model = joblib.load("model.pkl")  # Disk I/O every time!
    return model.predict(...)

# ✅ GOOD: Load once at startup (5ms latency)
model = joblib.load("model.pkl")  # Global, loaded once

@app.post("/predict")
def predict(txn: Transaction):
    return model.predict(...)  # Memory access
```

### **Pattern 2: Feature Store**

```python
# Problem: Model needs customer_avg_amount at inference
# But we only have current transaction

# ✅ Solution: Maintain feature cache
feature_cache = {
    "CUST_001": {"avg_amount": 120.0, "std": 45.0, "txn_count": 156},
    "CUST_002": {"avg_amount": 85.0, "std": 30.0, "txn_count": 89}
    # ... updated daily from transaction history
}

@app.post("/predict")
def predict(txn: Transaction):
    # Lookup customer features
    customer_features = feature_cache.get(txn.customer_id, {
        "avg_amount": 100.0,  # Global default
        "std": 50.0,
        "txn_count": 0
    })

    features = [
        txn.amount,
        customer_features["avg_amount"],
        abs(txn.amount - customer_features["avg_amount"]) / customer_features["std"],
        ...
    ]
    return model.predict([features])
```

### **Pattern 3: Monitoring**

```python
from prometheus_client import Counter, Histogram

# Define metrics
predictions_total = Counter(
    'fraud_predictions_total',
    'Total predictions',
    ['outcome']  # Labels: fraud/normal
)
prediction_latency = Histogram(
    'fraud_prediction_latency_seconds',
    'Prediction latency'
)

@app.post("/predict")
def predict(txn: Transaction):
    # Track latency
    with prediction_latency.time():
        result = model.predict(...)

    # Track outcome
    outcome = "fraud" if result > 0.5 else "normal"
    predictions_total.labels(outcome=outcome).inc()

    return result

# Metrics endpoint for Prometheus scraping
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

---

## 4️⃣ SYSTEM DESIGN

### **Latency Requirements**

```
┌─────────────────────────────────────────────────────────┐
│ Single Prediction:    <50ms p99                         │
│   - Validation:       1ms   (Pydantic)                  │
│   - Feature lookup:   2ms   (Redis cache)               │
│   - Inference:        5ms   (cached model)              │
│   - SHAP:            10ms   (cached explainer)          │
│   - Serialization:    2ms   (JSON response)             │
│   Total:            ~20ms   (✅ Under budget)           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Batch Prediction:     Async (no blocking)               │
│   - 10,000 rows:     ~30 seconds background             │
│   - Chunk size:      1,000 rows per batch               │
│   - Progress:        Updated every chunk                │
│   - Job storage:     SQLite (dev), Redis (prod)         │
└─────────────────────────────────────────────────────────┘
```

### **Scaling Strategy**

```
Stage 1: Single Server (0-100 RPS)
    ├─ FastAPI + Uvicorn
    ├─ Model in memory
    └─ SQLite for jobs

Stage 2: Horizontal Scaling (100-1000 RPS)
    ├─ Load balancer (Nginx)
    ├─ 3-5 FastAPI replicas
    ├─ Redis for feature cache + job queue
    └─ PostgreSQL for persistence

Stage 3: Microservices (1000+ RPS)
    ├─ Separate model serving (Triton)
    ├─ Feature service (Feast)
    ├─ Job queue (RabbitMQ + Celery)
    └─ Monitoring (Prometheus + Grafana)
```

---

## 5️⃣ METRICS & EVALUATION

### **Model Metrics**

```python
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ROC-AUC: Probability that model ranks fraud > normal
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.3f}")  # Target: >0.90

# Precision/Recall at threshold
y_pred = (y_proba > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"False Positives: {fp}")  # Legitimate blocked
print(f"False Negatives: {fn}")  # Fraud missed
```

**Fraud Detection Trade-offs**:
- **High Precision**: Few false alarms, but miss some fraud
- **High Recall**: Catch all fraud, but many false alarms
- **Balance**: F1 score or custom threshold based on cost

### **API Metrics**

```python
# Key metrics to track
Latency:
    - p50 (median): <10ms
    - p95: <30ms
    - p99: <50ms
    - p99.9: <100ms

Throughput:
    - Requests per second (RPS)
    - Target: 100-1000 RPS depending on scale

Error Rate:
    - 4xx errors (client): <1%
    - 5xx errors (server): <0.1%

Availability:
    - Uptime: 99.9% (SLA: "three nines")
    - Downtime: <45 minutes/month
```

---

## 6️⃣ DATA HANDLING

### **Imbalanced Data (Fraud: 3%, Normal: 97%)**

```python
# Strategy 1: SMOTE (Synthetic Minority Oversampling)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Original fraud count: {y_train.sum()}")
print(f"After SMOTE: {y_train_balanced.sum()}")

# Strategy 2: Class Weights
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    class_weight='balanced',  # Automatically adjust weights
    ...
)

# Strategy 3: Custom Threshold
# Instead of 0.5, use threshold that optimizes F1
from sklearn.metrics import f1_score

thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = [f1_score(y_val, y_proba > t) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
```

### **Feature Engineering**

```python
# Time-based features
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_night'] = df['hour'].isin(range(0, 6)).astype(int)

# Behavioral features
customer_stats = df.groupby('customer_id')['amount'].agg(['mean', 'std'])
df = df.merge(customer_stats, on='customer_id', suffixes=['', '_customer_avg'])
df['amount_deviation'] = (df['amount'] - df['amount_customer_avg']) / df['std']

# Velocity features
df['time_since_last_txn'] = df.groupby('customer_id')['timestamp'].diff().dt.total_seconds() / 60
```

---

## 7️⃣ DEPLOYMENT

### **Docker Setup**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 8000 8501

# Run both services
CMD ["sh", "-c", "uvicorn src.backend.app:app --host 0.0.0.0 --port 8000 & streamlit run src/frontend/app.py --server.port 8501"]
```

```bash
# Build and run
docker build -t fraud-detection .
docker run -p 8000:8000 -p 8501:8501 fraud-detection
```

### **CI/CD Pipeline**

```yaml
# .github/workflows/deploy.yml
name: Deploy ML Model

on:
  push:
    branches: [main]

jobs:
  test-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/ --cov=src

      - name: Train model
        run: python src/ml/train_sklearn.py

      - name: Build Docker image
        run: docker build -t fraud-api .

      - name: Deploy to production
        run: |
          docker push fraud-api:latest
          kubectl rollout restart deployment/fraud-api
```

---

## 8️⃣ DEBUGGING & TROUBLESHOOTING

### **Common Issues**

| Problem | Cause | Solution |
|---------|-------|----------|
| `Model training fails silently` | Softmax + CrossEntropyLoss | Remove softmax from forward() |
| `Optuna: DeprecationWarning` | suggest_loguniform deprecated | Use suggest_float(..., log=True) |
| `ModuleNotFoundError` | Missing __init__.py | Add empty __init__.py to packages |
| `Streamlit UI freezes` | time.sleep() blocks | Use st.rerun() or auto-refresh |
| `API latency >500ms` | Loading model per request | Load model once at startup |
| `SHAP values all zero` | Wrong class index | Use shap_values[1] for fraud class |
| `Predictions always 0.5` | Feature scaling mismatch | Use same scaler as training |

### **Debug Checklist**

```python
# 1. Verify model loaded correctly
assert model is not None
print(f"Model type: {type(model)}")
print(f"Model params: {model.get_params()}")

# 2. Check feature shapes
print(f"Training features: {X_train.shape}")
print(f"Test features: {X_test.shape}")
print(f"Inference features: {features.shape}")
assert X_train.shape[1] == features.shape[1], "Feature count mismatch!"

# 3. Validate predictions
y_proba = model.predict_proba(X_test)
assert y_proba.shape == (len(X_test), 2), "Wrong probability shape"
assert np.allclose(y_proba.sum(axis=1), 1.0), "Probabilities don't sum to 1"

# 4. Check for data leakage
print(f"Train AUC: {roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])}")
print(f"Test AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")
# If train AUC >> test AUC (e.g., 0.99 vs 0.75), suspect overfitting or leakage
```

---

## 9️⃣ INTERVIEW SCENARIOS

### **Q: "Walk me through your ML pipeline"**

**Answer**:
1. **Data**: Synthetic transaction generator creates 10K rows with 3% fraud
2. **Features**: Engineer 21 features (temporal, behavioral, network)
3. **Split**: 80/20 train/test with stratification (preserve fraud ratio)
4. **Balance**: SMOTE to oversample minority class (fraud)
5. **Train**: RandomForest baseline + PyTorch MLP via Skorch
6. **Tune**: Optuna Bayesian optimization (10 trials)
7. **Evaluate**: ROC-AUC, precision/recall, confusion matrix
8. **Explain**: SHAP values for top features
9. **Track**: MLflow logs params, metrics, artifacts
10. **Serve**: FastAPI endpoint with <50ms latency
11. **Monitor**: Prometheus metrics (latency, throughput, errors)

### **Q: "How do you handle model versioning?"**

**Answer**:
```python
# models/
#   sklearn_rf_v1.pkl
#   sklearn_rf_v2.pkl
#   model_registry.json

{
  "production": {
    "model_path": "models/sklearn_rf_v2.pkl",
    "version": "v2",
    "deployed_at": "2025-01-20T10:00:00",
    "metrics": {"roc_auc": 0.92, "precision": 0.88}
  },
  "staging": {
    "model_path": "models/pytorch_mlp_v1.pkl",
    "version": "v1",
    "metrics": {"roc_auc": 0.91, "precision": 0.87}
  }
}
```

Load model based on environment:
```python
import json

with open("models/model_registry.json") as f:
    registry = json.load(f)

env = os.getenv("ENV", "production")
model_path = registry[env]["model_path"]
model = joblib.load(model_path)
```

### **Q: "How do you ensure reproducibility?"**

**Answer**:
1. **Pin dependencies**: requirements.txt with exact versions
2. **Set random seeds**: `np.random.seed(42)`, `torch.manual_seed(42)`
3. **Version data**: Feature store with timestamps (features_v1_20250120.parquet)
4. **Track experiments**: MLflow logs code version, params, metrics
5. **Docker**: Containerize environment for consistent deployment
6. **Documentation**: Document data sources, preprocessing steps

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

---

## 🔟 PRODUCTION CHECKLIST

### **Before Deploying**

- [ ] Model performance validated (ROC-AUC >0.90)
- [ ] API latency tested (p99 <50ms)
- [ ] Error handling implemented (try/except, logging)
- [ ] Input validation (Pydantic schemas)
- [ ] Monitoring setup (Prometheus metrics)
- [ ] Logging configured (structured JSON logs)
- [ ] Health check endpoint (/health)
- [ ] Load testing completed (1000+ RPS)
- [ ] Security audit (no secrets in code)
- [ ] Documentation written (API docs, runbooks)

### **Post-Deployment**

- [ ] Monitor error rates (target: <0.1%)
- [ ] Track latency (p99 should stay <50ms)
- [ ] Watch for data drift (distribution shifts)
- [ ] Set up alerts (Slack/PagerDuty)
- [ ] Plan for model retraining (weekly/monthly)
- [ ] A/B test new models (challenger vs champion)
- [ ] Collect user feedback (false positive reports)
- [ ] Review SHAP explanations (ensure sensible)

---

## 💡 KEY TAKEAWAYS

1. **Skorch makes PyTorch sklearn-compatible** → Pipelines, GridSearch, Optuna
2. **CrossEntropyLoss expects logits** → Don't apply softmax in forward()
3. **Optuna >> GridSearch** → 10 trials vs 100+ for deep learning
4. **FastAPI + Pydantic = Type Safety** → Invalid requests fail early
5. **Streamlit reruns scripts** → Use st.session_state for persistence
6. **Load models once** → Startup not per-request (5ms vs 500ms)
7. **Background tasks for batches** → Don't block API
8. **SHAP explains predictions** → Compliance requirement
9. **MLflow tracks experiments** → Reproducibility + comparison
10. **Prometheus monitors production** → Metrics > logs

---

**Interview Mantra**: "I build production ML systems, not Jupyter notebooks. My models are fast (<50ms), explainable (SHAP), monitored (Prometheus), and continuously improving (MLflow + Optuna)."

---

**Remember**: You're not just a data scientist. You're a **full-stack ML engineer**.
