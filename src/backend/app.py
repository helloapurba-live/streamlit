"""
============================================================================
FastAPI Backend for Bank Transaction Anomaly Detection
============================================================================

LEARNING OBJECTIVES:
1. Build production-ready REST API with FastAPI
2. Implement single and batch prediction endpoints
3. Manage background jobs with SQLite job tracking
4. Integrate ML model loading and inference
5. Add Prometheus monitoring and health checks
6. Provide interactive API documentation with Swagger UI

THEORY:
--------
A production ML API requires:
- Request validation (Pydantic models ensure type safety)
- Asynchronous processing (background jobs for batch workloads)
- State management (SQLite for job tracking, in-memory for models)
- Error handling (graceful failures with informative messages)
- Monitoring (Prometheus metrics for latency, throughput, errors)
- Documentation (auto-generated OpenAPI/Swagger docs)

This backend provides:
1. POST /predict_single → Real-time prediction for one transaction
2. POST /predict_batch → Asynchronous batch processing (CSV upload)
3. GET /job_status/{job_id} → Poll batch job progress
4. GET /download_result/{job_id} → Download batch results
5. POST /agent_investigate → LLM-powered transaction investigation
6. GET /metrics → Prometheus metrics endpoint

ARCHITECTURE:
-------------
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Application                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │  Request     │    │   Model      │    │  Job Queue   │    │
│  │  Validation  │───▶│   Inference  │───▶│  Background  │    │
│  │  (Pydantic)  │    │   (sklearn/  │    │  Tasks       │    │
│  │              │    │    PyTorch)  │    │              │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           SQLite Job Tracker Database                │    │
│  │  jobs table: job_id, status, progress, result_path   │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐    │
│  │        Prometheus Metrics Collector                   │    │
│  │  - prediction_latency, batch_jobs_total, errors       │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
"""

import os
import sys
import uuid
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import shap

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class TransactionInput(BaseModel):
    """
    Single transaction input for real-time prediction.

    All fields are required for model inference.
    """
    customer_id: str = Field(..., example="CUST_000123")
    amount: float = Field(..., gt=0, example=127.45, description="Transaction amount in USD")
    merchant_category: str = Field(..., example="grocery")
    merchant_id: str = Field(..., example="MERCH_01234")
    latitude: float = Field(..., ge=-90, le=90, example=37.7749)
    longitude: float = Field(..., ge=-180, le=180, example=-122.4194)
    is_online: int = Field(..., ge=0, le=1, example=0)
    hour: int = Field(..., ge=0, le=23, example=14)
    day_of_week: int = Field(..., ge=0, le=6, example=2)
    is_weekend: int = Field(..., ge=0, le=1, example=0)
    is_night: int = Field(..., ge=0, le=1, example=0)

    @validator('merchant_category')
    def validate_category(cls, v):
        valid_categories = [
            'grocery', 'gas_transport', 'restaurant', 'entertainment',
            'online_shopping', 'bills_utilities', 'health_fitness', 'travel'
        ]
        if v not in valid_categories:
            raise ValueError(f"Invalid category. Must be one of {valid_categories}")
        return v


class PredictionResponse(BaseModel):
    """Response for single prediction request."""
    transaction_id: str
    customer_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud_predicted: bool
    model_version: str
    explanation: Dict[str, Any]
    processing_time_ms: float


class BatchJobResponse(BaseModel):
    """Response when batch job is submitted."""
    job_id: str
    status: str
    message: str
    submitted_at: str


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: float = Field(..., ge=0, le=100)
    submitted_at: str
    completed_at: Optional[str] = None
    total_records: Optional[int] = None
    processed_records: Optional[int] = None
    result_path: Optional[str] = None
    error_message: Optional[str] = None


class InvestigateRequest(BaseModel):
    """Request for agent-based investigation."""
    customer_id: str = Field(..., example="CUST_000123")
    transaction_ids: Optional[List[str]] = None
    query: str = Field(..., example="Why was this account flagged?")


class InvestigateResponse(BaseModel):
    """Response from investigation agent."""
    customer_id: str
    summary: str
    risk_score: float
    key_findings: List[str]
    recommended_actions: List[str]


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Counters
prediction_counter = Counter(
    'fraud_predictions_total',
    'Total number of fraud predictions',
    ['model_version', 'prediction']
)
batch_job_counter = Counter(
    'batch_jobs_total',
    'Total number of batch jobs',
    ['status']
)
api_errors = Counter(
    'api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type']
)

# Histograms
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['endpoint']
)


# ============================================================================
# JOB TRACKER (SQLite Database)
# ============================================================================

class JobTracker:
    """
    Manages background job state in SQLite database.

    Jobs table schema:
    - job_id (TEXT PRIMARY KEY)
    - status (TEXT): 'pending', 'running', 'completed', 'failed'
    - progress (REAL): 0-100
    - submitted_at (TEXT)
    - completed_at (TEXT)
    - total_records (INTEGER)
    - processed_records (INTEGER)
    - result_path (TEXT)
    - error_message (TEXT)
    """

    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create jobs table if not exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0,
                submitted_at TEXT NOT NULL,
                completed_at TEXT,
                total_records INTEGER,
                processed_records INTEGER DEFAULT 0,
                result_path TEXT,
                error_message TEXT
            )
        """)
        conn.commit()
        conn.close()

    def create_job(self, job_id: str, total_records: int) -> None:
        """Create new job entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO jobs (job_id, status, submitted_at, total_records)
            VALUES (?, ?, ?, ?)
        """, (job_id, 'pending', datetime.now().isoformat(), total_records))
        conn.commit()
        conn.close()

    def update_job(self, job_id: str, **kwargs) -> None:
        """Update job fields."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [job_id]

        cursor.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)
        conn.commit()
        conn.close()

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Retrieve job details."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None


# ============================================================================
# MODEL LOADER
# ============================================================================

class ModelManager:
    """
    Manages ML model loading and caching.

    Loads models from disk and keeps them in memory for fast inference.
    Supports both sklearn and PyTorch models via joblib.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.explainers = {}
        self._load_models()

    def _load_models(self):
        """Load all available models from model directory."""
        print("Loading models...")

        # Try to load sklearn model
        sklearn_path = self.model_dir / "sklearn_rf_model.pkl"
        if sklearn_path.exists():
            try:
                self.models['sklearn_rf'] = joblib.load(sklearn_path)
                print(f"✅ Loaded sklearn model from {sklearn_path}")

                # Load SHAP explainer if available
                explainer_path = self.model_dir / "sklearn_rf_explainer.pkl"
                if explainer_path.exists():
                    self.explainers['sklearn_rf'] = joblib.load(explainer_path)
                    print(f"✅ Loaded SHAP explainer")
            except Exception as e:
                print(f"❌ Failed to load sklearn model: {e}")

        # Try to load PyTorch model
        pytorch_path = self.model_dir / "pytorch_mlp_model.pkl"
        if pytorch_path.exists():
            try:
                self.models['pytorch_mlp'] = joblib.load(pytorch_path)
                print(f"✅ Loaded PyTorch model from {pytorch_path}")
            except Exception as e:
                print(f"❌ Failed to load PyTorch model: {e}")

        # Fallback: create dummy model if no models found
        if not self.models:
            print("⚠️  No trained models found. Using dummy model for demo.")
            self.models['dummy'] = self._create_dummy_model()

    def _create_dummy_model(self):
        """Create a simple dummy model for demonstration."""
        class DummyModel:
            def predict_proba(self, X):
                # Simple rule: flag high amounts as fraud
                probs = []
                for row in X:
                    amount = row[0] if isinstance(row, (list, np.ndarray)) else row
                    fraud_prob = min(0.95, max(0.05, amount / 1000))
                    probs.append([1 - fraud_prob, fraud_prob])
                return np.array(probs)

            def predict(self, X):
                probs = self.predict_proba(X)
                return (probs[:, 1] > 0.5).astype(int)

        return DummyModel()

    def get_model(self, model_name: str = None):
        """Get model by name or default model."""
        if model_name and model_name in self.models:
            return self.models[model_name]
        # Return first available model
        return list(self.models.values())[0] if self.models else None

    def get_explainer(self, model_name: str = None):
        """Get SHAP explainer for model."""
        if model_name and model_name in self.explainers:
            return self.explainers[model_name]
        return None


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Bank Transaction Anomaly Detection API",
    description="Production ML API for real-time and batch fraud detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize global components
job_tracker = JobTracker(db_path=str(PROJECT_ROOT / "jobs.db"))
model_manager = ModelManager(model_dir=str(PROJECT_ROOT / "models"))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def prepare_features(transaction: TransactionInput) -> np.ndarray:
    """
    Convert transaction input to feature array for model.

    Order must match training features:
    [amount, is_online, hour, day_of_week, is_weekend, is_night, ...]
    """
    features = np.array([[
        transaction.amount,
        transaction.is_online,
        transaction.hour,
        transaction.day_of_week,
        transaction.is_weekend,
        transaction.is_night,
        # Add derived features with dummy values for single prediction
        0.0,  # time_since_last_transaction (unknown for single)
        transaction.amount,  # customer_avg_amount (use current as proxy)
        0.0,  # customer_std_amount
        0.0,  # amount_deviation
        0.5,  # merchant_risk_score (neutral)
        1.0,  # distance_from_home (dummy)
    ]])
    return features


def explain_prediction(
    model,
    explainer,
    features: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Generate explanation for prediction using SHAP.

    Returns dictionary with top contributing features.
    """
    if explainer is None:
        # Fallback: use feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            top_indices = np.argsort(importance)[-5:][::-1]
            return {
                "method": "feature_importance",
                "top_features": [
                    {"feature": feature_names[i], "importance": float(importance[i])}
                    for i in top_indices
                ]
            }
        else:
            return {"method": "none", "message": "No explainer available"}

    # Use SHAP explainer
    try:
        shap_values = explainer.shap_values(features)
        # For binary classification, take positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Fraud class

        # Get top 5 contributors
        abs_shap = np.abs(shap_values[0])
        top_indices = np.argsort(abs_shap)[-5:][::-1]

        return {
            "method": "shap",
            "top_features": [
                {
                    "feature": feature_names[i],
                    "shap_value": float(shap_values[0][i]),
                    "feature_value": float(features[0][i])
                }
                for i in top_indices
            ]
        }
    except Exception as e:
        return {"method": "error", "message": str(e)}


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Bank Transaction Anomaly Detection API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": len(model_manager.models),
        "endpoints": {
            "docs": "/docs",
            "predict_single": "/predict_single",
            "predict_batch": "/predict_batch",
            "job_status": "/job_status/{job_id}",
            "download_result": "/download_result/{job_id}",
            "investigate": "/agent_investigate",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_available": list(model_manager.models.keys())
    }


@app.post("/predict_single", response_model=PredictionResponse)
async def predict_single(transaction: TransactionInput):
    """
    ENDPOINT: Single Transaction Prediction

    Performs real-time fraud prediction for one transaction.

    REQUEST BODY (JSON):
    {
        "customer_id": "CUST_000123",
        "amount": 127.45,
        "merchant_category": "grocery",
        "merchant_id": "MERCH_01234",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "is_online": 0,
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_night": 0
    }

    RESPONSE:
    {
        "transaction_id": "uuid-...",
        "customer_id": "CUST_000123",
        "fraud_probability": 0.15,
        "is_fraud_predicted": false,
        "model_version": "sklearn_rf",
        "explanation": {
            "method": "shap",
            "top_features": [
                {"feature": "amount", "shap_value": 0.02, "feature_value": 127.45},
                ...
            ]
        },
        "processing_time_ms": 12.5
    }
    """
    start_time = time.time()

    try:
        # Prepare features
        features = prepare_features(transaction)

        # Get model and predict
        model = model_manager.get_model('sklearn_rf')
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model available for inference"
            )

        proba = model.predict_proba(features)[0]
        fraud_probability = float(proba[1])
        is_fraud = fraud_probability > 0.5

        # Generate explanation
        feature_names = [
            'amount', 'is_online', 'hour', 'day_of_week', 'is_weekend', 'is_night',
            'time_since_last', 'customer_avg', 'customer_std', 'amount_dev',
            'merchant_risk', 'distance_home'
        ]
        explainer = model_manager.get_explainer('sklearn_rf')
        explanation = explain_prediction(model, explainer, features, feature_names)

        # Record metrics
        processing_time = (time.time() - start_time) * 1000
        prediction_latency.labels(endpoint='predict_single').observe(time.time() - start_time)
        prediction_counter.labels(
            model_version='sklearn_rf',
            prediction='fraud' if is_fraud else 'legitimate'
        ).inc()

        return PredictionResponse(
            transaction_id=str(uuid.uuid4()),
            customer_id=transaction.customer_id,
            fraud_probability=fraud_probability,
            is_fraud_predicted=is_fraud,
            model_version='sklearn_rf',
            explanation=explanation,
            processing_time_ms=processing_time
        )

    except Exception as e:
        api_errors.labels(endpoint='predict_single', error_type=type(e).__name__).inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_batch", response_model=BatchJobResponse)
async def predict_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    ENDPOINT: Batch Transaction Prediction

    Uploads CSV file for asynchronous batch processing.

    REQUEST:
    - file: CSV with columns matching TransactionInput schema

    RESPONSE:
    {
        "job_id": "batch_uuid-...",
        "status": "pending",
        "message": "Batch job submitted successfully",
        "submitted_at": "2025-01-19T10:30:00"
    }

    WORKFLOW:
    1. Upload CSV → job created → returns job_id
    2. Poll /job_status/{job_id} for progress
    3. When complete, download from /download_result/{job_id}
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only CSV files are supported"
            )

        # Save uploaded file
        job_id = f"batch_{uuid.uuid4()}"
        upload_dir = PROJECT_ROOT / "uploads"
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{job_id}.csv"

        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)

        # Read to get row count
        df = pd.read_csv(file_path)
        total_records = len(df)

        # Create job in tracker
        job_tracker.create_job(job_id, total_records)

        # Schedule background processing
        background_tasks.add_task(process_batch_job, job_id, file_path)

        batch_job_counter.labels(status='submitted').inc()

        return BatchJobResponse(
            job_id=job_id,
            status='pending',
            message=f"Batch job submitted successfully. Processing {total_records} records.",
            submitted_at=datetime.now().isoformat()
        )

    except Exception as e:
        api_errors.labels(endpoint='predict_batch', error_type=type(e).__name__).inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit batch job: {str(e)}"
        )


def process_batch_job(job_id: str, file_path: Path):
    """
    Background task: Process batch predictions.

    Updates job progress in database as it processes each row.
    """
    try:
        # Update status to running
        job_tracker.update_job(job_id, status='running', progress=0.0)

        # Load data
        df = pd.read_csv(file_path)
        total = len(df)

        # Get model
        model = model_manager.get_model('sklearn_rf')
        if model is None:
            raise Exception("No model available")

        # Prepare features (simplified - assumes all columns are present)
        # In production, use proper feature engineering pipeline
        feature_cols = [
            'amount', 'is_online', 'hour', 'day_of_week', 'is_weekend', 'is_night'
        ]

        # Add dummy columns if not present
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Process in batches of 100
        results = []
        batch_size = 100

        for i in range(0, total, batch_size):
            batch_df = df.iloc[i:i+batch_size]

            # Predict
            X = batch_df[feature_cols].values
            # Add dummy features to match model expectations (12 features total)
            X_padded = np.hstack([X, np.zeros((len(X), 6))])

            proba = model.predict_proba(X_padded)
            predictions = (proba[:, 1] > 0.5).astype(int)
            fraud_scores = proba[:, 1]

            # Add results
            batch_df['fraud_probability'] = fraud_scores
            batch_df['is_fraud_predicted'] = predictions
            results.append(batch_df)

            # Update progress
            processed = min(i + batch_size, total)
            progress = (processed / total) * 100
            job_tracker.update_job(
                job_id,
                progress=progress,
                processed_records=processed
            )

        # Combine results
        result_df = pd.concat(results, ignore_index=True)

        # Save result
        result_dir = PROJECT_ROOT / "results"
        result_dir.mkdir(exist_ok=True)
        result_path = result_dir / f"{job_id}_results.csv"
        result_df.to_csv(result_path, index=False)

        # Update job as completed
        job_tracker.update_job(
            job_id,
            status='completed',
            progress=100.0,
            completed_at=datetime.now().isoformat(),
            result_path=str(result_path)
        )

        batch_job_counter.labels(status='completed').inc()

    except Exception as e:
        # Mark job as failed
        job_tracker.update_job(
            job_id,
            status='failed',
            error_message=str(e)
        )
        batch_job_counter.labels(status='failed').inc()


@app.get("/job_status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    ENDPOINT: Get Batch Job Status

    Poll this endpoint to check progress of batch job.

    RESPONSE:
    {
        "job_id": "batch_uuid-...",
        "status": "running",
        "progress": 45.5,
        "submitted_at": "2025-01-19T10:30:00",
        "completed_at": null,
        "total_records": 1000,
        "processed_records": 455,
        "result_path": null,
        "error_message": null
    }

    Status values: 'pending', 'running', 'completed', 'failed'
    """
    job = job_tracker.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    return JobStatusResponse(**job)


@app.get("/download_result/{job_id}")
async def download_result(job_id: str):
    """
    ENDPOINT: Download Batch Results

    Download CSV file with predictions.

    Returns FileResponse with CSV attachment.
    """
    job = job_tracker.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    if job['status'] != 'completed':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job status is '{job['status']}'. Only completed jobs can be downloaded."
        )

    result_path = Path(job['result_path'])
    if not result_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result file not found"
        )

    return FileResponse(
        path=result_path,
        media_type='text/csv',
        filename=f"{job_id}_results.csv"
    )


@app.post("/agent_investigate", response_model=InvestigateResponse)
async def agent_investigate(request: InvestigateRequest):
    """
    ENDPOINT: Agent-Based Investigation

    Uses LLM-powered agent to investigate customer account and transactions.

    This is a simplified demo. In production, this would:
    1. Retrieve transaction history from database
    2. Load LLM (local or API)
    3. Generate investigation report with RAG
    4. Provide actionable recommendations

    REQUEST:
    {
        "customer_id": "CUST_000123",
        "transaction_ids": ["txn_1", "txn_2"],
        "query": "Why was this account flagged?"
    }

    RESPONSE:
    {
        "customer_id": "CUST_000123",
        "summary": "Account shows unusual spending patterns...",
        "risk_score": 0.75,
        "key_findings": ["Finding 1", "Finding 2"],
        "recommended_actions": ["Action 1", "Action 2"]
    }
    """
    # Simulated investigation (replace with actual LLM in production)
    time.sleep(0.5)  # Simulate processing

    return InvestigateResponse(
        customer_id=request.customer_id,
        summary=f"Investigation of {request.customer_id}: {request.query}. "
                f"Analysis shows moderate risk based on recent transaction patterns. "
                f"3 transactions flagged in the last 24 hours with amounts exceeding "
                f"baseline by 5-8x. Geographic spread across 2 states detected.",
        risk_score=0.65,
        key_findings=[
            "3 high-value transactions ($800-$1200) in 24 hours",
            "Transactions in multiple states (CA, NY)",
            "New merchant categories not seen in 90-day history",
            "Velocity anomaly: 5 transactions in 2-hour window"
        ],
        recommended_actions=[
            "Contact customer for verification",
            "Place temporary hold on card",
            "Review transaction details with fraud team",
            "Monitor for additional suspicious activity"
        ]
    )


@app.get("/metrics")
async def metrics():
    """
    ENDPOINT: Prometheus Metrics

    Exposes metrics in Prometheus format for monitoring.

    Metrics include:
    - fraud_predictions_total
    - batch_jobs_total
    - api_errors_total
    - prediction_latency_seconds
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    print("\n" + "="*70)
    print("🚀 STARTING FRAUD DETECTION API")
    print("="*70)
    print(f"📦 Models loaded: {len(model_manager.models)}")
    print(f"📊 Job tracker initialized: {PROJECT_ROOT / 'jobs.db'}")
    print(f"📡 API docs: http://localhost:8000/docs")
    print("="*70 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


"""
============================================================================
LINE-BY-LINE EXPLANATION
============================================================================

Lines 1-62: Module docstring with learning objectives, theory, and ASCII architecture

Lines 64-80: Imports
    - FastAPI: web framework
    - Pydantic: request/response validation
    - pandas/numpy: data processing
    - prometheus_client: monitoring metrics
    - joblib: model loading

Lines 89-129: TransactionInput Pydantic model
    - Defines required fields for prediction
    - Includes validation (amount > 0, lat/lon ranges)
    - Custom validator for merchant_category
    - Field examples for API docs

Lines 132-182: Response Pydantic models
    - PredictionResponse: single prediction result
    - BatchJobResponse: batch submission confirmation
    - JobStatusResponse: job progress tracking
    - InvestigateRequest/Response: agent endpoint schemas

Lines 187-219: Prometheus metrics setup
    - Counters: track totals (predictions, jobs, errors)
    - Histograms: track latencies
    - Labels: allow filtering by model version, status, etc.

Lines 224-294: JobTracker class
    - SQLite database for job state persistence
    - _init_db: creates jobs table
    - create_job: insert new job
    - update_job: update job fields (status, progress)
    - get_job: retrieve job details

Lines 299-354: ModelManager class
    - Loads models from disk on startup
    - Caches in memory for fast inference
    - Supports sklearn and PyTorch models
    - _create_dummy_model: fallback if no models found
    - get_model/get_explainer: retrieve by name

Lines 359-371: FastAPI app initialization
    - Title, description, version for docs
    - Auto-generated docs at /docs and /redoc

Lines 373-375: Initialize global components
    - job_tracker: SQLite database
    - model_manager: loaded models

Lines 381-398: prepare_features function
    - Converts Pydantic model to numpy array
    - Order must match training feature order
    - Adds dummy values for derived features (since we don't have history)

Lines 401-447: explain_prediction function
    - Uses SHAP explainer if available
    - Falls back to feature_importances_ for sklearn
    - Returns top 5 contributing features
    - Handles both SHAP values and basic importance

Lines 453-469: Root and health check endpoints
    - /: API info and endpoint list
    - /health: health check for k8s/monitoring

Lines 472-550: /predict_single endpoint
    - Accepts TransactionInput JSON
    - Prepares features
    - Runs inference with sklearn model
    - Generates SHAP explanation
    - Records Prometheus metrics
    - Returns PredictionResponse with fraud probability and explanation

Lines 553-611: /predict_batch endpoint
    - Accepts CSV file upload
    - Creates job in tracker
    - Schedules background task
    - Returns job_id immediately (async)

Lines 614-695: process_batch_job background task
    - Runs in separate thread
    - Processes CSV in batches of 100
    - Updates job progress in database
    - Saves results to CSV
    - Handles errors and marks job failed if exception

Lines 698-729: /job_status/{job_id} endpoint
    - Queries job tracker database
    - Returns current status and progress
    - Used for polling by frontend

Lines 732-761: /download_result/{job_id} endpoint
    - Checks job is completed
    - Returns FileResponse with CSV
    - Used to download batch predictions

Lines 764-816: /agent_investigate endpoint
    - Simulated LLM-powered investigation
    - In production: would use RAG with transaction history
    - Returns risk score and recommended actions

Lines 819-827: /metrics endpoint
    - Exposes Prometheus metrics
    - Used by Prometheus scraper or monitoring tools

Lines 833-842: startup_event
    - Prints startup info to console
    - Runs when FastAPI app starts

Lines 845-847: Main execution block
    - Runs uvicorn server if script executed directly

============================================================================
SAMPLE REQUEST & RESPONSE
============================================================================

REQUEST TO /predict_single (POST):
curl -X POST "http://localhost:8000/predict_single" \
     -H "Content-Type: application/json" \
     -d '{
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
         }'

RESPONSE FROM /predict_single:
{
  "transaction_id": "8f3a2b1c-4d5e-6f7g-8h9i-0j1k2l3m4n5o",
  "customer_id": "CUST_000123",
  "fraud_probability": 0.78,
  "is_fraud_predicted": true,
  "model_version": "sklearn_rf",
  "explanation": {
    "method": "shap",
    "top_features": [
      {
        "feature": "amount",
        "shap_value": 0.25,
        "feature_value": 850.0
      },
      {
        "feature": "is_night",
        "shap_value": 0.12,
        "feature_value": 1.0
      },
      {
        "feature": "is_online",
        "shap_value": 0.08,
        "feature_value": 1.0
      }
    ]
  },
  "processing_time_ms": 15.3
}

============================================================================
POWERSHELL COMMANDS TO RUN
============================================================================

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run FastAPI backend
cd C:\Users\YourName\ml-fraud-dashboard
uvicorn src.backend.app:app --host 0.0.0.0 --port 8000 --reload

# Expected output:
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
# INFO:     Started reloader process [PID]
# 🚀 STARTING FRAUD DETECTION API
# ======================================================================
# 📦 Models loaded: 1
# 📊 Job tracker initialized: C:\...\jobs.db
# 📡 API docs: http://localhost:8000/docs
# ======================================================================

# Test health endpoint
curl http://localhost:8000/health

# Test single prediction (PowerShell)
$body = @{
    customer_id = "CUST_000123"
    amount = 850.00
    merchant_category = "online_shopping"
    merchant_id = "MERCH_09876"
    latitude = 40.7128
    longitude = -74.0060
    is_online = 1
    hour = 23
    day_of_week = 5
    is_weekend = 1
    is_night = 1
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict_single" -Method POST -Body $body -ContentType "application/json"

# Open API docs in browser
start http://localhost:8000/docs

============================================================================
EXERCISE
============================================================================

TASK: Add a new endpoint /model_info that returns detailed information about
      the currently loaded model(s).

REQUIREMENTS:
1. Endpoint: GET /model_info
2. Return model name, version, training date, performance metrics
3. Include feature list and feature importance if available
4. Add caching to avoid repeated file reads

SOLUTION:
----------
@app.get("/model_info")
async def model_info():
    \"\"\"Get information about loaded models.\"\"\"
    model_info_list = []

    for model_name, model in model_manager.models.items():
        info = {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "features": []
        }

        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_names = [
                'amount', 'is_online', 'hour', 'day_of_week',
                'is_weekend', 'is_night', 'time_since_last',
                'customer_avg', 'customer_std', 'amount_dev',
                'merchant_risk', 'distance_home'
            ]
            importance = model.feature_importances_
            info['features'] = [
                {"name": name, "importance": float(imp)}
                for name, imp in zip(feature_names, importance)
            ]

        model_info_list.append(info)

    return {"models": model_info_list}

============================================================================
"""
