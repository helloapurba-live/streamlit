# 🎯 Top 50 ML Dashboard Interview Questions & Answers

## Complete Job-Ready Interview Guide

---

## 📋 Quick Navigation

**Part 1: Foundation (Q1-Q15)**
- Conceptual understanding of tech stack
- Core implementation patterns
- Basic code examples

**Part 2: Advanced Topics (Q16-Q30)**
- System design considerations
- Performance optimization
- Production patterns

**Part 3: MLOps & Deployment (Q31-Q45)**
- Experiment tracking
- Model versioning
- Monitoring and observability

**Part 4: Troubleshooting (Q46-Q50)**
- Debugging common issues
- Performance optimization
- Real-world scenarios

---

## PART 1: FOUNDATION (Q1-Q15)

### Q1: Why Skorch over pure PyTorch?

**Answer**: Skorch bridges PyTorch and scikit-learn, giving **sklearn API** (`.fit()`, `.predict_proba()`) to PyTorch models. This enables:
- Pipeline compatibility (`StandardScaler → NeuralNetClassifier`)
- Hyperparameter tuning (GridSearchCV, Optuna)
- Cross-validation with `cross_val_score()`

**Code**:
```python
net = NeuralNetClassifier(MyPyTorchModel, lr=0.001, max_epochs=50)
pipeline = Pipeline([('scaler', StandardScaler()), ('net', net)])
pipeline.fit(X, y)  # Works like sklearn!
```

---

### Q2: Critical PyTorch mistake with CrossEntropyLoss?

**Answer**: Applying **softmax in forward()** is wrong.

**Why**: `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`. It expects **logits** (raw scores), not probabilities.

```python
# ❌ WRONG
def forward(self, x):
    return F.softmax(self.fc(x), dim=-1)

# ✅ CORRECT
def forward(self, x):
    return self.fc(x)  # Return logits
```

---

### Q3: Optuna vs GridSearchCV?

**Answer**:
- **GridSearch**: Exhaustive (tests all combinations) → 100+ trials
- **Optuna**: Bayesian optimization (learns from trials) → 10 trials

**Key**: Use `log=True` for learning rate:
```python
lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)  # Log scale
```

**Saves 90% compute time** for deep learning.

---

### Q4: Streamlit execution model?

**Answer**: **Reruns entire script** on every interaction (unlike Flask's route handlers).

**Implications**:
- Use `@st.cache_data` for expensive operations
- Use `st.session_state` to persist values
- Use `st.form()` to batch inputs

```python
if 'count' not in st.session_state:
    st.session_state.count = 0

if st.button("Click"):
    st.session_state.count += 1  # Survives reruns
```

---

### Q5: What is SHAP and why use it?

**Answer**: SHAP (SHapley Additive exPlanations) provides **per-prediction** feature contributions.

**vs Feature Importance**:
- Feature Importance: "Feature X is globally important"
- SHAP: "For THIS prediction, feature X contributed +0.25"

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)[1]  # Fraud class
# Shows: amount (+0.25), hour (+0.12), distance (+0.08)
```

**Why**: Compliance, trust, debugging spurious correlations.

---

### Q6: Handle imbalanced data (3% fraud)?

**Answer**: Three strategies:

**1. SMOTE** (Synthetic Minority Oversampling):
```python
from imblearn.over_sampling import SMOTE
X_balanced, y_balanced = SMOTE().fit_resample(X_train, y_train)
```

**2. Class Weights**:
```python
model = RandomForestClassifier(class_weight='balanced')
```

**3. Custom Threshold**:
```python
# Find threshold that optimizes F1
optimal_threshold = 0.3  # Instead of 0.5
y_pred = (y_proba > optimal_threshold).astype(int)
```

---

### Q7: MLflow vs Model Versioning?

**Answer**:
- **MLflow**: Tracks experiments (params, metrics, artifacts)
- **Model Versioning**: Manages deployment (v1 in prod, v2 in staging)

**Together**:
```python
with mlflow.start_run():
    mlflow.log_params({"lr": 0.001})
    mlflow.log_metric("roc_auc", 0.92)
    mlflow.sklearn.log_model(model, "model")

# Promote to production
client.transition_model_version_stage(name="fraud", version=2, stage="Production")
```

---

### Q8: Load model once at startup?

**Answer**: **500ms disk I/O** per request vs **5ms memory access**.

```python
# ❌ BAD: Load per request (505ms latency)
@app.post("/predict")
def predict(txn):
    model = joblib.load("model.pkl")  # 500ms
    return model.predict(...)  # 5ms

# ✅ GOOD: Load once (5ms latency)
model = joblib.load("model.pkl")  # Load globally

@app.post("/predict")
def predict(txn):
    return model.predict(...)  # 5ms
```

---

### Q9: Streamlit vs React for stakeholders?

**Answer**:

"**Streamlit** = PowerPoint for data apps (30 lines → full dashboard)
**React** = Photoshop for web apps (300+ lines, full control)"

**Use Streamlit for**: Internal tools, analysts, data science demos
**Use React for**: Customer-facing apps, complex UX, mobile

---

### Q10: CAP Theorem of ML?

**Answer**: Choose 2 of 3:
1. **Speed** (<50ms latency)
2. **Accuracy** (>95% ROC-AUC)
3. **Interpretability** (explainable)

**Examples**:
- **Fraud detection**: Speed + Accuracy (use SHAP post-hoc)
- **Loan approval**: Interpretability + Accuracy (regulatory)
- **Research**: Accuracy + Interpretability (trade latency)

---

### Q11: FastAPI request validation?

**Answer**: Pydantic provides **compile-time** type safety.

```python
from pydantic import BaseModel, Field

class Transaction(BaseModel):
    amount: float = Field(gt=0, le=1000000)  # 0 < amount ≤ 1M
    category: str = Field(example="grocery")

@app.post("/predict")
def predict(txn: Transaction):  # Already validated!
    return model.predict([txn.amount, ...])
```

Invalid requests → **422 error** before reaching code.

---

### Q12: Optuna hyperparameter tuning?

**Answer**:

```python
def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)  # Log scale!
    hidden = trial.suggest_int('hidden_dim', 32, 256)

    net = NeuralNetClassifier(Model, lr=lr, module__hidden_dim=hidden)
    return cross_val_score(net, X, y, cv=3).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # Finds optimum in 10 trials
```

**Key**: TPE (Tree-structured Parzen Estimator) learns which regions work.

---

### Q13: Streamlit session state?

**Answer**: Persist data across script reruns.

```python
if 'job_id' not in st.session_state:
    st.session_state.job_id = None

if st.button("Submit"):
    st.session_state.job_id = submit_job()
    st.rerun()  # Refresh page

# job_id survives rerun
if st.session_state.job_id:
    status = check_status(st.session_state.job_id)
```

---

### Q14: SHAP visualization in Streamlit?

**Answer**:

```python
import shap
import streamlit as st

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)[1]  # Fraud class

# Matplotlib integration
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)
```

**Types**: Waterfall, Force plot, Summary plot

---

### Q15: Background job processing?

**Answer**: Use **BackgroundTasks** for async processing.

```python
@app.post("/predict_batch")
async def predict_batch(file: UploadFile, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    bg.add_task(process_batch, job_id, file)  # Runs in background
    return {"job_id": job_id, "status": "pending"}

def process_batch(job_id, file):
    # Process in chunks, update progress in database
    for chunk in chunks:
        predictions = model.predict(chunk)
        update_progress(job_id, progress)
```

**For scale**: Use Redis + Celery instead of SQLite.

---

## PART 2: SYSTEM DESIGN (Q16-Q30)

### Q16: How to design a low-latency ML API?

**Answer**: **5-layer optimization**:

**1. Model Caching** (500ms → 5ms)
```python
model = joblib.load("model.pkl")  # Load once at startup
```

**2. Feature Cache** (Redis for customer features)
```python
customer_features = redis.get(f"customer:{customer_id}")
```

**3. Batch Inference** (Process multiple requests together)
```python
if len(queue) >= 32:  # Batch size
    predictions = model.predict(queue)
```

**4. Model Quantization** (Reduce model size)
```python
# Convert float32 → int8 (4x smaller, faster)
quantized_model = torch.quantization.quantize_dynamic(model)
```

**5. Async Processing** (Non-blocking I/O)
```python
@app.post("/predict")
async def predict(txn: Transaction):
    return await model_service.predict(txn)
```

**Result**: <50ms p99 latency

---

### Q17: Handle training-serving skew?

**Answer**: **Problem**: Model trained on `customer_avg_amount` but unavailable at inference.

**Solution 1: Feature Store**
```python
feature_store = {
    "CUST_001": {"avg_amount": 120.0, "std": 45.0},
    # Updated daily from transaction history
}

@app.post("/predict")
def predict(txn):
    features = [txn.amount, feature_store[txn.customer_id]["avg_amount"], ...]
```

**Solution 2: Real-time Computation**
```python
# Query transaction history in real-time
recent_txns = db.query(f"SELECT * FROM txns WHERE customer_id='{txn.customer_id}' LIMIT 100")
customer_avg = recent_txns['amount'].mean()
```

**Solution 3: Low-Confidence Flag**
```python
if using_dummy_features:
    return {"prediction": 0.7, "confidence": "LOW"}
```

---

### Q18: Scale to 1000+ requests/second?

**Answer**: **Architecture evolution**:

**Stage 1: Single Server (0-100 RPS)**
```
FastAPI + Uvicorn
Model in memory
SQLite for jobs
```

**Stage 2: Horizontal Scaling (100-1000 RPS)**
```
Load Balancer (Nginx)
├─ FastAPI replica 1
├─ FastAPI replica 2
├─ FastAPI replica 3
Redis (feature cache + job queue)
PostgreSQL (persistence)
```

**Stage 3: Microservices (1000+ RPS)**
```
API Gateway
├─ Model Serving (Triton Inference Server)
├─ Feature Service (Feast)
├─ Job Queue (RabbitMQ + Celery)
└─ Monitoring (Prometheus + Grafana)
```

---

### Q19: Monitor model performance in production?

**Answer**: **Three-layer monitoring**:

**1. API Metrics (Prometheus)**
```python
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions', ['outcome'])
prediction_latency = Histogram('prediction_latency_seconds', 'Latency')

@app.post("/predict")
@prediction_latency.time()
def predict(txn):
    result = model.predict(...)
    prediction_counter.labels(outcome="fraud" if result > 0.5 else "normal").inc()
    return result
```

**2. Model Metrics (Daily Batch)**
```python
# Compute metrics on labeled data
daily_auc = roc_auc_score(y_true_yesterday, y_pred_yesterday)
if daily_auc < 0.85:  # Alert if performance degrades
    send_alert("Model performance dropped to {daily_auc:.2f}")
```

**3. Data Drift (Evidently AI)**
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train, current_data=X_prod_last_week)
```

---

### Q20: Implement A/B testing for models?

**Answer**: **Shadow mode** → **Canary** → **Full rollout**

**1. Shadow Mode** (both models run, only v1 serves)
```python
@app.post("/predict")
def predict(txn):
    # Production model
    result_v1 = model_v1.predict(txn)

    # Shadow model (log but don't serve)
    result_v2 = model_v2.predict(txn)
    log_shadow_prediction(txn, result_v1, result_v2)

    return result_v1  # Only return v1
```

**2. Canary** (5% traffic to v2)
```python
@app.post("/predict")
def predict(txn):
    if hash(txn.customer_id) % 100 < 5:  # 5% of users
        return model_v2.predict(txn)
    else:
        return model_v1.predict(txn)
```

**3. Full Rollout** (after validation)
```python
# If v2 ROC-AUC ≥ v1 and no errors
model = model_v2  # Switch globally
```

---

### Q21: Design a feature store?

**Answer**: **Requirements**:
1. **Versioned**: Features change over time
2. **Low-latency**: <10ms lookup for inference
3. **Consistent**: Same features for training and serving

**Architecture**:
```
┌─────────────────────────────────────────┐
│ Feature Store                           │
├─────────────────────────────────────────┤
│                                         │
│ Offline Store (Training)                │
│ ├─ Parquet files on S3                  │
│ └─ features_v1_20250120.parquet         │
│                                         │
│ Online Store (Serving)                  │
│ ├─ Redis (key-value cache)              │
│ └─ customer:CUST_001 → {avg: 120.0}    │
│                                         │
│ Feature Registry                        │
│ └─ Metadata (schema, lineage, owner)   │
└─────────────────────────────────────────┘
```

**Code**:
```python
# Training: Read from offline store
features = pd.read_parquet(f"s3://features/features_v1_{date}.parquet")

# Serving: Read from online store (Redis)
customer_features = redis.get(f"customer:{customer_id}")
```

**Tools**: Feast, Tecton, AWS Feature Store

---

### Q22: Handle model versioning and rollback?

**Answer**: **Model Registry Pattern**:

```python
# models/model_registry.json
{
  "production": {
    "path": "models/fraud_v2.pkl",
    "version": "v2",
    "deployed_at": "2025-01-20T10:00:00",
    "metrics": {"roc_auc": 0.92}
  },
  "staging": {
    "path": "models/fraud_v3.pkl",
    "version": "v3",
    "metrics": {"roc_auc": 0.93}
  },
  "previous": {
    "path": "models/fraud_v1.pkl",
    "version": "v1",
    "metrics": {"roc_auc": 0.90}
  }
}
```

**Loading**:
```python
import json

with open("models/model_registry.json") as f:
    registry = json.load(f)

env = os.getenv("MODEL_ENV", "production")
model_path = registry[env]["path"]
model = joblib.load(model_path)
```

**Rollback**:
```bash
# If v2 has issues, rollback to v1
jq '.production = .previous' model_registry.json > temp.json
mv temp.json model_registry.json
# Restart service (auto-loads v1)
```

---

### Q23: Implement circuit breaker pattern?

**Answer**: **Prevent cascading failures** when model service is down.

```python
from collections import deque
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = deque(maxlen=failure_threshold)
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.opened_at = None

    def call(self, func, *args, **kwargs):
        # OPEN: Circuit is broken, reject immediately
        if self.state == "OPEN":
            if datetime.now() - self.opened_at > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"  # Try again
            else:
                raise Exception("Circuit is OPEN, service unavailable")

        try:
            result = func(*args, **kwargs)

            # Success: Reset failures if in HALF_OPEN
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures.clear()

            return result

        except Exception as e:
            self.failures.append(datetime.now())

            # Too many failures: Open circuit
            if len(self.failures) >= self.failure_threshold:
                self.state = "OPEN"
                self.opened_at = datetime.now()

            raise e

# Usage
circuit_breaker = CircuitBreaker()

@app.post("/predict")
def predict(txn):
    try:
        return circuit_breaker.call(model.predict, [txn.features])
    except Exception:
        # Fallback: Return conservative prediction
        return {"fraud_probability": 0.5, "confidence": "LOW", "source": "fallback"}
```

---

### Q24: Design for horizontal scalability?

**Answer**: **Stateless services + External state**

**❌ Bad (Stateful)**:
```python
# Global state in API server
cache = {}  # Lost when server restarts

@app.post("/predict")
def predict(txn):
    if txn.customer_id in cache:
        return cache[txn.customer_id]
    # ...
```

**✅ Good (Stateless)**:
```python
# External state in Redis
import redis
cache = redis.Redis()

@app.post("/predict")
def predict(txn):
    cached = cache.get(txn.customer_id)
    if cached:
        return json.loads(cached)
    # ...
```

**Benefits**:
- Can run 10 API replicas behind load balancer
- Any replica can handle any request
- Restart doesn't lose state

---

### Q25: Implement rate limiting?

**Answer**: **Token bucket algorithm** via middleware.

```python
from fastapi import Request, HTTPException
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        minute_ago = now - 60

        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]

        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False

        # Record request
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(requests_per_minute=100)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Get client ID (IP or API key)
    client_id = request.client.host

    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later."
        )

    return await call_next(request)
```

**Production**: Use **Redis** for distributed rate limiting.

---

### Q26: Handle database connection pooling?

**Answer**: **Connection pooling** reduces overhead of creating connections.

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Create engine with connection pool
engine = create_engine(
    "postgresql://user:pass@localhost/fraud_db",
    poolclass=QueuePool,
    pool_size=10,  # Max 10 connections
    max_overflow=5,  # Allow 5 extra during spikes
    pool_timeout=30,  # Wait 30s for connection
    pool_recycle=3600  # Recycle connections after 1 hour
)

@app.on_event("startup")
def startup():
    # Test connection on startup
    with engine.connect() as conn:
        conn.execute("SELECT 1")

@app.post("/predict")
def predict(txn):
    # Get connection from pool (fast)
    with engine.connect() as conn:
        result = conn.execute(
            "SELECT * FROM customer_features WHERE customer_id = ?",
            (txn.customer_id,)
        )
    # Connection returned to pool
```

**Benefits**:
- Reuse connections (avoid TCP handshake overhead)
- Limit max connections (prevent DB overload)
- Auto-recovery from connection failures

---

### Q27: Implement health checks for Kubernetes?

**Answer**: **Liveness** and **Readiness** probes.

```python
from fastapi import status

@app.get("/health/liveness")
def liveness():
    """
    Liveness probe: Is the service running?

    Kubernetes restarts pod if this fails.
    """
    return {"status": "alive"}

@app.get("/health/readiness")
def readiness():
    """
    Readiness probe: Is the service ready to handle traffic?

    Kubernetes removes from load balancer if this fails.
    """
    # Check dependencies
    checks = {
        "model_loaded": model is not None,
        "database_connected": check_db_connection(),
        "redis_connected": check_redis_connection()
    }

    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "checks": checks}
        )

def check_db_connection():
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except:
        return False
```

**Kubernetes Deployment**:
```yaml
livenessProbe:
  httpGet:
    path: /health/liveness
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/readiness
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
```

---

### Q28: Design for disaster recovery?

**Answer**: **Backup + Replication + Failover**

**1. Model Backups**
```bash
# Automated daily backups
0 2 * * * aws s3 sync /models/ s3://backups/models/$(date +\%Y-\%m-\%d)/
```

**2. Database Replication**
```
Primary DB (Write)
    ├─ Replica 1 (Read)
    ├─ Replica 2 (Read)
    └─ Failover DB (Standby)
```

**3. Multi-Region Deployment**
```
US-East (Primary)
    ├─ API replicas: 5
    ├─ DB: Primary
    └─ Redis: Cluster

US-West (Failover)
    ├─ API replicas: 2
    ├─ DB: Read replica
    └─ Redis: Replica

# Route53 health checks automatically failover
```

**4. Regular DR Drills**
```python
# Monthly test: Simulate primary region failure
def disaster_recovery_drill():
    # 1. Stop primary region services
    # 2. Promote replica DB to primary
    # 3. Update DNS to point to failover region
    # 4. Verify service is operational
    # 5. Rollback
```

---

### Q29: Implement request tracing?

**Answer**: **Distributed tracing** with correlation IDs.

```python
import uuid
from contextvars import ContextVar

# Thread-local storage for request ID
request_id_var: ContextVar[str] = ContextVar('request_id', default='')

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    # Generate or extract request ID
    request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    request_id_var.set(request_id)

    # Add to response headers
    response = await call_next(request)
    response.headers['X-Request-ID'] = request_id

    return response

@app.post("/predict")
def predict(txn: Transaction):
    request_id = request_id_var.get()

    # Log with request ID
    logger.info(f"[{request_id}] Processing prediction for {txn.customer_id}")

    try:
        result = model.predict(...)
        logger.info(f"[{request_id}] Prediction: {result}")
        return result
    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        raise

# Now you can trace entire request flow
# [abc-123] Processing prediction for CUST_001
# [abc-123] Feature extraction: 12 features
# [abc-123] Model inference: 5ms
# [abc-123] SHAP computation: 10ms
# [abc-123] Prediction: 0.78
```

**Production**: Use **OpenTelemetry** or **Jaeger** for distributed tracing.

---

### Q30: Design API versioning strategy?

**Answer**: **URL versioning** with backward compatibility.

```python
from fastapi import FastAPI

app_v1 = FastAPI()
app_v2 = FastAPI()

# V1: Original API
@app_v1.post("/predict")
def predict_v1(amount: float, category: str):
    return {"fraud_probability": model_v1.predict([[amount]])[0]}

# V2: Enhanced API with more features
@app_v2.post("/predict")
def predict_v2(txn: TransactionV2):  # More detailed schema
    return {
        "fraud_probability": model_v2.predict(...)  # Better model
        "confidence": "HIGH",
        "explanation": shap_values  # New feature
    }

# Main app with versioning
app = FastAPI()
app.mount("/v1", app_v1)
app.mount("/v2", app_v2)

# Deprecation notice
@app_v1.get("/")
def v1_root():
    return {
        "version": "v1",
        "deprecated": True,
        "sunset_date": "2025-12-31",
        "migration_guide": "/docs/v1-to-v2"
    }
```

**Benefits**:
- Old clients keep working (v1)
- New clients use improved API (v2)
- Gradual migration period

---

## PART 3: MLOPS (Q31-Q45)

### Q31: Set up MLflow experiment tracking?

**Answer**:

```python
import mlflow
import mlflow.sklearn

# Configure tracking URI
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

# Set experiment
mlflow.set_experiment("fraud_detection")

# Training loop
with mlflow.start_run(run_name="random_forest_v1"):
    # Log parameters
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 10,
        "class_weight": "balanced"
    })

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Log metrics
    mlflow.log_metrics({
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    })

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log artifacts
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr)
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")

# View in UI
# mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

---

### Q32: Implement feature versioning?

**Answer**: **Timestamped Parquet files** + metadata.

```python
from datetime import datetime
import pandas as pd
import json

def save_features(df, version=None):
    """Save features with version metadata."""
    if version is None:
        version = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save features
    feature_path = f"feature_store/features_{version}.parquet"
    df.to_parquet(feature_path)

    # Save metadata
    metadata = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "n_rows": len(df),
        "n_features": len(df.columns),
        "columns": list(df.columns),
        "schema": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

    metadata_path = f"feature_store/metadata_{version}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved features version {version}")
    return version

def load_features(version="latest"):
    """Load features by version."""
    if version == "latest":
        # Find most recent version
        import glob
        files = glob.glob("feature_store/features_*.parquet")
        version = max(files).split('_')[-1].replace('.parquet', '')

    feature_path = f"feature_store/features_{version}.parquet"
    metadata_path = f"feature_store/metadata_{version}.json"

    df = pd.read_parquet(feature_path)

    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"📂 Loaded features version {version}")
    print(f"   Rows: {metadata['n_rows']:,}")
    print(f"   Features: {metadata['n_features']}")

    return df, metadata
```

---

### Q33: Create model card for documentation?

**Answer**:

```markdown
# Model Card: Fraud Detection Random Forest

## Model Details
- **Model Type**: Random Forest Classifier
- **Version**: v2.1
- **Training Date**: 2025-01-20
- **Owner**: ML Team (ml-team@company.com)
- **Framework**: scikit-learn 1.4.0

## Intended Use
- **Primary Use**: Real-time fraud detection for credit card transactions
- **Out-of-Scope**: ACH transfers, international transactions, B2B payments
- **Users**: Fraud analysts, customer service reps

## Training Data
- **Source**: Internal transaction database (2023-2024)
- **Size**: 1,000,000 transactions
- **Distribution**: 97% legitimate, 3% fraud
- **Features**: 21 (temporal, behavioral, network)
- **Sampling**: SMOTE oversampling for fraud class

## Performance Metrics
- **ROC-AUC**: 0.92 (test set)
- **Precision**: 0.88 (fraud class)
- **Recall**: 0.85 (fraud class)
- **F1 Score**: 0.86
- **Latency**: <10ms p99

## Fairness & Bias
- **Protected Attributes**: Age, gender, zip code (not used)
- **Fairness Metrics**: Equal opportunity difference <0.05 across demographics
- **Bias Mitigation**: Regular audits, SHAP explanations

## Limitations
- **Known Issues**:
  - Lower recall for transactions <$10 (38% vs 85% overall)
  - Degrades on new merchant categories (requires retraining)
- **Edge Cases**: International transactions may have higher false positive rate

## Monitoring
- **Metrics Tracked**: Daily ROC-AUC, precision, recall
- **Alerts**: Email if AUC drops below 0.85
- **Retraining**: Monthly or when AUC < 0.88

## Ethical Considerations
- **Impact**: Blocking legitimate transactions causes customer friction
- **Mitigation**: Human review for borderline cases (0.45-0.55 probability)
- **Appeals Process**: Customers can appeal blocks via support

## References
- Training code: `src/ml/train_sklearn.py`
- MLflow run: `mlruns/1/abc123`
- Model file: `models/sklearn_rf_v2.pkl`
```

---

### Q34: Implement automated retraining pipeline?

**Answer**: **Scheduled retraining** with performance validation.

```python
# retrain_pipeline.py
import schedule
import time
from datetime import datetime, timedelta

def retrain_model():
    """
    Automated retraining pipeline.

    Steps:
    1. Load recent data (last 30 days)
    2. Generate features
    3. Train new model
    4. Validate performance
    5. Deploy if better than current
    """
    print(f"\n{'='*70}")
    print(f"STARTING RETRAINING: {datetime.now()}")
    print(f"{'='*70}\n")

    # 1. Load data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    df = load_data(start_date, end_date)
    print(f"✅ Loaded {len(df):,} transactions")

    # 2. Generate features
    features_version = save_features(df)
    X = df[feature_cols]
    y = df['is_fraud']

    # 3. Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    model.fit(X_train, y_train)

    # 4. Validate performance
    y_proba = model.predict_proba(X_test)[:, 1]
    new_auc = roc_auc_score(y_test, y_proba)

    # Load current production model
    current_model = joblib.load("models/production.pkl")
    current_auc = roc_auc_score(y_test, current_model.predict_proba(X_test)[:, 1])

    print(f"\nPerformance Comparison:")
    print(f"  Current model AUC: {current_auc:.4f}")
    print(f"  New model AUC:     {new_auc:.4f}")

    # 5. Deploy if better
    if new_auc >= current_auc * 0.98:  # Allow 2% tolerance
        print(f"\n✅ New model is better! Deploying...")

        # Save new model with version
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        joblib.dump(model, f"models/fraud_{version}.pkl")

        # Update production symlink
        import shutil
        shutil.copy(f"models/fraud_{version}.pkl", "models/production.pkl")

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_metric("roc_auc", new_auc)
            mlflow.log_param("version", version)
            mlflow.sklearn.log_model(model, "model")

        # Send notification
        send_slack_message(f"✅ Model retrained and deployed. AUC: {new_auc:.4f}")

    else:
        print(f"\n⚠️ New model is not better. Keeping current model.")
        send_slack_message(f"⚠️ Retraining complete but model not deployed (AUC: {new_auc:.4f} vs {current_auc:.4f})")

# Schedule: Every Sunday at 2 AM
schedule.every().sunday.at("02:00").do(retrain_model)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(60)
```

**Production**: Use **Airflow** or **Prefect** instead of `schedule`.

---

### Q35: Monitor data drift?

**Answer**: Use **Evidently** to detect distribution shifts.

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import pandas as pd

def check_data_drift(reference_data, current_data, threshold=0.1):
    """
    Check if current data has drifted from reference.

    Args:
        reference_data: Training data distribution
        current_data: Production data (last week)
        threshold: Max allowed drift score

    Returns:
        drift_report: Dictionary with drift metrics
    """
    # Create drift report
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])

    report.run(reference_data=reference_data, current_data=current_data)

    # Extract metrics
    drift_metrics = report.as_dict()

    # Check for significant drift
    dataset_drift = drift_metrics['metrics'][0]['result']['dataset_drift']
    drifted_features = [
        feature['column_name']
        for feature in drift_metrics['metrics'][0]['result']['drift_by_columns'].values()
        if feature.get('drift_detected', False)
    ]

    print(f"\n{'='*70}")
    print(f"DATA DRIFT REPORT")
    print(f"{'='*70}")
    print(f"Dataset drift detected: {dataset_drift}")
    print(f"Drifted features ({len(drifted_features)}): {drifted_features}")

    # Alert if drift detected
    if dataset_drift or len(drifted_features) > 3:
        alert_message = f"""
        ⚠️ DATA DRIFT ALERT

        Dataset drift: {dataset_drift}
        Drifted features: {drifted_features}

        Recommended action: Review model performance and consider retraining.
        """
        send_alert(alert_message)

    # Save HTML report
    report.save_html("drift_report.html")

    return {
        "dataset_drift": dataset_drift,
        "drifted_features": drifted_features,
        "report_path": "drift_report.html"
    }

# Usage: Run daily
X_train_reference = pd.read_parquet("feature_store/features_training.parquet")
X_prod_last_week = load_production_data(days=7)

drift_report = check_data_drift(X_train_reference, X_prod_last_week)
```

---

### Q36: Implement model explainability dashboard?

**Answer**: **Streamlit app** for SHAP visualizations.

```python
import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd

st.title("🔍 Model Explainability Dashboard")

# Load model and explainer
@st.cache_resource
def load_components():
    model = joblib.load("models/production.pkl")
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_components()

# Sidebar: Choose analysis type
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Global Importance", "Individual Prediction", "Cohort Comparison"]
)

if analysis_type == "Global Importance":
    st.header("Global Feature Importance")

    # Load sample data
    X_test = pd.read_parquet("data/test_sample.parquet")

    with st.spinner("Computing SHAP values..."):
        shap_values = explainer.shap_values(X_test)[1]  # Fraud class

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    st.pyplot(fig)

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)

elif analysis_type == "Individual Prediction":
    st.header("Explain Single Prediction")

    # Input transaction
    amount = st.number_input("Amount", value=850.0)
    hour = st.slider("Hour", 0, 23, 23)
    # ... more inputs

    if st.button("Explain"):
        features = prepare_features(amount, hour, ...)
        shap_values = explainer.shap_values(features)[1]

        # Waterfall plot
        fig = plt.figure()
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value[1],
            shap_values[0],
            features[0],
            show=False
        )
        st.pyplot(fig)

elif analysis_type == "Cohort Comparison":
    st.header("Compare Feature Importance Across Cohorts")

    cohort = st.selectbox("Cohort", ["High Value", "Low Value", "Online", "In-Person"])

    # Filter data by cohort
    X_cohort = filter_by_cohort(X_test, cohort)
    shap_cohort = explainer.shap_values(X_cohort)[1]

    # Plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_cohort, X_cohort, plot_type='bar', show=False)
    st.pyplot(fig)
```

---

### Q37: Set up CI/CD for ML models?

**Answer**: **GitHub Actions** workflow.

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Training and Deployment

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest tests/ -v --cov=src

      - name: Generate synthetic data
        run: |
          python data/generate_synthetic_transactions.py

      - name: Train model
        run: |
          python src/ml/train_sklearn.py
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}

      - name: Validate model performance
        run: |
          python scripts/validate_model.py
          # Exit code 1 if performance < threshold

      - name: Build Docker image
        run: |
          docker build -t fraud-api:${{ github.sha }} .

      - name: Push to registry
        run: |
          docker push fraud-api:${{ github.sha }}
          docker tag fraud-api:${{ github.sha }} fraud-api:latest
          docker push fraud-api:latest

      - name: Deploy to staging
        run: |
          kubectl set image deployment/fraud-api fraud-api=fraud-api:${{ github.sha }} -n staging

      - name: Run smoke tests
        run: |
          python tests/smoke_tests.py --env staging

      - name: Deploy to production (manual approval)
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: trstringer/manual-approval@v1
        with:
          approvers: ml-team,devops-team

      - name: Production deployment
        run: |
          kubectl set image deployment/fraud-api fraud-api=fraud-api:${{ github.sha }} -n production

      - name: Slack notification
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'ML model deployed to production'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

### Q38: Implement model performance monitoring?

**Answer**: **Prometheus + Grafana** dashboard.

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Model performance metrics
model_predictions = Counter(
    'model_predictions_total',
    'Total predictions',
    ['model_version', 'outcome']
)

model_accuracy = Gauge(
    'model_accuracy',
    'Model accuracy (last hour)',
    ['model_version']
)

prediction_confidence = Histogram(
    'prediction_confidence',
    'Distribution of prediction confidences',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Compute metrics periodically
def compute_model_metrics():
    """Compute and update model performance metrics."""
    # Get predictions from last hour
    predictions = get_recent_predictions(hours=1)

    if len(predictions) > 0:
        # Compute accuracy (if labels available)
        if 'true_label' in predictions.columns:
            accuracy = (predictions['prediction'] == predictions['true_label']).mean()
            model_accuracy.labels(model_version='v2').set(accuracy)

        # Log prediction distribution
        for _, row in predictions.iterrows():
            model_predictions.labels(
                model_version='v2',
                outcome='fraud' if row['prediction'] == 1 else 'normal'
            ).inc()

            prediction_confidence.observe(row['probability'])

# Run every hour
schedule.every(1).hours.do(compute_model_metrics)
```

**Grafana Dashboard** (`grafana/dashboard.json`):
```json
{
  "panels": [
    {
      "title": "Model Accuracy (Last 24h)",
      "targets": [
        {
          "expr": "model_accuracy{model_version='v2'}"
        }
      ]
    },
    {
      "title": "Predictions by Outcome",
      "targets": [
        {
          "expr": "rate(model_predictions_total[5m])"
        }
      ]
    },
    {
      "title": "Prediction Latency p99",
      "targets": [
        {
          "expr": "histogram_quantile(0.99, prediction_latency_seconds_bucket)"
        }
      ]
    }
  ]
}
```

---

### Q39: Handle model serving at scale (1000+ RPS)?

**Answer**: **Triton Inference Server** for GPU batching.

```python
# 1. Export model to ONNX
import torch
import torch.onnx

dummy_input = torch.randn(1, 12)  # Batch size 1, 12 features
torch.onnx.export(
    pytorch_model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# 2. Triton model repository structure
# models/
#   fraud_detector/
#     config.pbtxt
#     1/
#       model.onnx

# config.pbtxt
"""
name: "fraud_detector"
platform: "onnxruntime_onnx"
max_batch_size: 256
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 12 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }
]
instance_group [
  {
    count: 4
    kind: KIND_GPU
  }
]
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32, 64, 128, 256 ]
  max_queue_delay_microseconds: 5000
}
"""

# 3. Client code
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

def predict_batch(transactions):
    # Prepare input
    input_data = np.array([prepare_features(txn) for txn in transactions], dtype=np.float32)

    inputs = [
        httpclient.InferInput("input", input_data.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [
        httpclient.InferRequestedOutput("output")
    ]

    # Infer
    response = client.infer("fraud_detector", inputs, outputs=outputs)

    # Parse output
    predictions = response.as_numpy("output")
    return predictions[:, 1]  # Fraud probabilities

# Benefits:
# - GPU batching (process 256 requests together)
# - Dynamic batching (groups requests automatically)
# - Multi-model serving
# - Can handle 10,000+ RPS
```

---

### Q40: Implement shadow mode deployment?

**Answer**: **Run both models, compare results, serve only production**.

```python
@app.post("/predict")
async def predict(txn: Transaction):
    # Production model (serve this)
    result_prod = model_prod.predict_proba([txn.features])[0][1]

    # Shadow model (log but don't serve)
    result_shadow = model_shadow.predict_proba([txn.features])[0][1]

    # Log comparison
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "transaction_id": txn.transaction_id,
        "customer_id": txn.customer_id,
        "features": txn.features,
        "production_prediction": result_prod,
        "shadow_prediction": result_shadow,
        "difference": abs(result_prod - result_shadow)
    }

    # Store for analysis
    log_shadow_comparison(comparison)

    # Alert if large difference
    if abs(result_prod - result_shadow) > 0.3:
        send_alert(f"Large prediction difference for {txn.transaction_id}")

    # Return only production result
    return {"fraud_probability": result_prod}

def analyze_shadow_performance():
    """
    Weekly analysis of shadow model vs production.

    Metrics:
    - Mean absolute difference
    - Cases where shadow would change decision
    - True label comparison (if available)
    """
    comparisons = load_shadow_comparisons(days=7)

    # MAE
    mae = comparisons['difference'].mean()

    # Decision changes (crossing 0.5 threshold)
    prod_decisions = (comparisons['production_prediction'] > 0.5).astype(int)
    shadow_decisions = (comparisons['shadow_prediction'] > 0.5).astype(int)
    decision_changes = (prod_decisions != shadow_decisions).sum()

    print(f"Shadow Model Analysis (Last 7 Days):")
    print(f"  Total predictions: {len(comparisons):,}")
    print(f"  Mean absolute difference: {mae:.4f}")
    print(f"  Decision changes: {decision_changes:,} ({decision_changes/len(comparisons)*100:.2f}%)")

    # If shadow performs better, promote it
    if has_labels():
        shadow_auc = roc_auc_score(comparisons['true_label'], comparisons['shadow_prediction'])
        prod_auc = roc_auc_score(comparisons['true_label'], comparisons['production_prediction'])

        if shadow_auc > prod_auc:
            print(f"\n✅ Shadow model outperforms (AUC: {shadow_auc:.3f} vs {prod_auc:.3f})")
            print(f"   Promoting shadow to production...")
            promote_shadow_to_production()
```

---

## PART 4: DEBUGGING (Q41-Q50)

### Q41: Debug "Model training fails silently"?

**Answer**: Check **forward() returns logits, not softmax**.

```python
# Test
import torch
model = FraudMLP()
x = torch.randn(1, 12)
output = model(x)

print(f"Output shape: {output.shape}")  # Should be (1, 2)
print(f"Output values: {output}")  # Should be raw scores (can be negative)
print(f"Sum to 1?: {output.softmax(dim=-1).sum()}")  # Softmax should sum to 1

# If output already sums to ~1.0, you have softmax in forward()
if output.sum().item() > 0.99 and output.sum().item() < 1.01:
    print("❌ ERROR: forward() is applying softmax!")
```

---

### Q42: Debug "Optuna gives DeprecationWarning"?

**Answer**: Update to `suggest_float(..., log=True)`.

```python
# ❌ OLD (deprecated in Optuna 3.x)
lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

# ✅ NEW
lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
```

---

### Q43: Debug "ModuleNotFoundError: No module named 'src'"?

**Answer**: Add `__init__.py` files to all package directories.

```bash
# Create package structure
touch src/__init__.py
touch src/ml/__init__.py
touch src/backend/__init__.py
touch src/frontend/__init__.py
touch src/utils/__init__.py
```

---

### Q44: Debug "Streamlit UI freezes during polling"?

**Answer**: `time.sleep()` blocks the UI. Use **st.rerun()** or **auto-refresh**.

```python
# ❌ BAD: Blocks UI
while status != 'completed':
    time.sleep(5)  # UI frozen for 5 seconds
    status = check_status()

# ✅ GOOD: Use button-based refresh
if st.button("Refresh Status"):
    st.rerun()

# ✅ BETTER: Auto-refresh with st.empty()
status_placeholder = st.empty()

for _ in range(60):  # Max 60 polls
    status = check_status()
    status_placeholder.write(f"Status: {status}")

    if status == 'completed':
        break

    time.sleep(5)
    # Note: UI still freezes but updates between polls
```

---

### Q45: Debug "API latency >500ms"?

**Answer**: Check if **model loaded per request**.

```python
import time

# Profile endpoint
@app.post("/predict")
def predict(txn: Transaction):
    start = time.time()

    # Check where time is spent
    t1 = time.time()
    model = load_model()  # ← This might be the problem
    print(f"Model load: {(time.time() - t1)*1000:.1f}ms")

    t2 = time.time()
    features = extract_features(txn)
    print(f"Feature extraction: {(time.time() - t2)*1000:.1f}ms")

    t3 = time.time()
    result = model.predict([features])
    print(f"Inference: {(time.time() - t3)*1000:.1f}ms")

    print(f"Total: {(time.time() - start)*1000:.1f}ms")
    return result

# If "Model load" is >100ms, move to global scope
model = load_model()  # Load once at startup
```

---

### Q46: Debug "SHAP values all zero"?

**Answer**: Using **wrong class index** for binary classification.

```python
# For binary classification, shap_values is a list
shap_values = explainer.shap_values(X)

# ❌ WRONG: Uses both classes (averages to zero)
print(shap_values)  # List of 2 arrays

# ✅ CORRECT: Use fraud class (index 1)
shap_values_fraud = shap_values[1]
print(shap_values_fraud)  # Now you see values
```

---

### Q47: Debug "Predictions always 0.5"?

**Answer**: **Feature scaling mismatch** between training and serving.

```python
# Training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)

# ❌ WRONG: Forgot to save scaler
# Serving
X_new = [[850, 23, 1, ...]]  # Unscaled
predictions = model.predict_proba(X_new)  # Wrong distribution!

# ✅ CORRECT: Use same scaler
import joblib
joblib.dump(scaler, "scaler.pkl")

# Serving
scaler = joblib.load("scaler.pkl")
X_new_scaled = scaler.transform([[850, 23, 1, ...]])
predictions = model.predict_proba(X_new_scaled)
```

**Or use sklearn Pipeline**:
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)

# Serving (auto-scales)
predictions = pipeline.predict_proba(X_new)
```

---

### Q48: Debug "MLflow database locked"?

**Answer**: **SQLite lock** from multiple processes.

```bash
# Kill all MLflow processes
taskkill /IM mlflow.exe /F  # Windows
pkill -9 mlflow  # Linux

# Remove lock files
rm mlruns/.mlflow.db-shm
rm mlruns/.mlflow.db-wal

# Restart
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

**Production**: Use **PostgreSQL** instead of SQLite.

---

### Q49: Debug "Batch job stuck at 50%"?

**Answer**: Check **error in process_batch()** function.

```python
def process_batch(job_id, file_path):
    try:
        # ... processing code

        # Log progress
        print(f"[{job_id}] Processing chunk {i}/{total}")

    except Exception as e:
        # Mark as failed
        update_job(job_id, status='failed', error_message=str(e))

        # Log full traceback
        import traceback
        print(f"ERROR in job {job_id}:")
        print(traceback.format_exc())

# Check database for error
import sqlite3
conn = sqlite3.connect('jobs.db')
cursor = conn.cursor()
cursor.execute("SELECT error_message FROM jobs WHERE job_id = ?", (job_id,))
error = cursor.fetchone()
print(error)
```

---

### Q50: How would you debug production model degradation?

**Answer**: **5-step diagnostic process**:

**1. Check Data Drift**
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train, current_data=X_prod_last_week)

# If drift detected → Retrain model
```

**2. Analyze Recent Predictions**
```python
# Check for sudden pattern changes
recent_preds = load_predictions(days=7)
recent_preds.groupby('date')['fraud_probability'].agg(['mean', 'std']).plot()

# Check feature distributions
recent_preds[feature_cols].describe()
```

**3. Check for Label Feedback**
```python
# If you have labels (from investigations)
labeled_data = load_labeled_recent_data()
current_auc = roc_auc_score(labeled_data['true_label'], labeled_data['prediction'])

print(f"Current AUC: {current_auc:.3f}")
print(f"Training AUC: 0.92")  # Original performance

if current_auc < 0.85:
    print("⚠️ Performance degraded! Investigate further.")
```

**4. SHAP Analysis on Recent Data**
```python
# Check if model is using features differently
recent_shap = explainer.shap_values(X_prod_recent)[1]
original_shap = explainer.shap_values(X_train_sample)[1]

# Compare feature importance
recent_importance = np.abs(recent_shap).mean(axis=0)
original_importance = np.abs(original_shap).mean(axis=0)

import pandas as pd
pd.DataFrame({
    'feature': feature_names,
    'recent_importance': recent_importance,
    'original_importance': original_importance,
    'change': recent_importance - original_importance
}).sort_values('change', key=abs, ascending=False)
```

**5. Check for Adversarial Patterns**
```python
# Fraudsters adapt to model
# Check if fraud is concentrating in low-score region

frauds_by_score = labeled_data[labeled_data['true_label'] == 1].groupby(
    pd.cut(labeled_data['prediction'], bins=[0, 0.3, 0.5, 0.7, 1.0])
)['prediction'].count()

print("Fraud distribution by model score:")
print(frauds_by_score)

# If many frauds in <0.3 bucket → Model is being gamed
```

**Action Plan**:
- Data drift → Retrain immediately
- Performance drop → Rollback to previous model
- Adversarial patterns → Add new features, retrain
- No clear cause → Shadow mode new model, collect more data

---

## 🎓 FINAL TIPS

### Interview Preparation Strategy

**1. Memorize Key Patterns** (15 minutes before interview):
- Model caching (load once at startup)
- Skorch forward() returns logits
- Optuna uses `log=True` not `suggest_loguniform`
- SHAP values[1] for fraud class

**2. Practice Drawing**:
- System architecture (API → Model → Database)
- Data flow (Request → Validation → Inference → Response)
- Scaling evolution (Single server → Replicas → Microservices)

**3. Prepare Stories**:
- "I optimized API latency from 500ms to 5ms by..."
- "I detected data drift using Evidently and retrained the model..."
- "I implemented A/B testing with shadow mode deployment..."

**4. Know Your Metrics**:
- **Latency**: <50ms p99
- **ROC-AUC**: >0.90
- **Uptime**: 99.9% ("three nines")
- **Error Rate**: <0.1%

---

### Common Interview Red Flags to Avoid

❌ "I use GridSearch for hyperparameter tuning" (Too slow for deep learning)
✅ "I use Optuna's Bayesian optimization, converges in 10 trials vs 100+"

❌ "I load the model on every request" (500ms latency)
✅ "I load the model once at startup, cache in memory for <5ms latency"

❌ "I can't explain my model's predictions" (Black box)
✅ "I use SHAP values to explain individual predictions for compliance"

❌ "I don't monitor model performance" (Naive)
✅ "I track ROC-AUC daily and alert if it drops below threshold"

---

### The Ultimate Answer Template

When asked "How would you [build/optimize/debug] X?":

**1. Clarify Requirements**
"First, I'd ask about scale (RPS?), latency requirements (p99?), and availability (SLA?)..."

**2. Start Simple**
"I'd start with a simple solution: load model at startup, FastAPI endpoint, SQLite for jobs..."

**3. Show Scaling Path**
"To scale to 1000 RPS, I'd add Redis for caching, horizontal replicas, and Triton for GPU batching..."

**4. Add Monitoring**
"I'd instrument with Prometheus metrics (latency, throughput, errors) and set up alerts..."

**5. Discuss Trade-offs**
"The trade-off is complexity vs performance. For <100 RPS, simple is fine. For 1000+, we need the full stack..."

---

**You're now ready for any production ML systems interview!** 🚀

Remember: **Production ML is 90% engineering, 10% modeling**. Show you understand the 90%.
