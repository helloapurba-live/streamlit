from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime
import joblib

# ---------------------------------------------------------
# Database Setup
# ---------------------------------------------------------
DATABASE_URL = "sqlite:///./iris.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionHistory(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sepal_length = Column(Float)
    sepal_width = Column(Float)
    petal_length = Column(Float)
    petal_width = Column(Float)
    predicted_species = Column(String)
    confidence = Column(Float)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Training model...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    models["clf"] = clf
    models["target_names"] = iris.target_names
    print("Model trained and loaded.")
    yield
    models.clear()

app = FastAPI(
    title="Iris Classification API",
    description="A simple API to predict Iris flower species.",
    version="2.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionOut(BaseModel):
    species: str
    probabilities: dict[str, float]

class HistoryOut(BaseModel):
    id: int
    timestamp: datetime
    sepal_length: float
    predicted_species: str
    confidence: float

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Iris API v2. Go to /docs for Swagger UI."}

@app.post("/predict", response_model=PredictionOut)
def predict(data: IrisInput, db: Session = Depends(get_db)):
    if "clf" not in models:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    
    # Inference
    prediction_idx = models["clf"].predict(input_data)[0]
    prediction_proba = models["clf"].predict_proba(input_data)[0]
    predicted_species = models["target_names"][prediction_idx]
    
    # Log to DB
    max_prob = float(max(prediction_proba))
    db_record = PredictionHistory(
        sepal_length=data.sepal_length,
        sepal_width=data.sepal_width,
        petal_length=data.petal_length,
        petal_width=data.petal_width,
        predicted_species=predicted_species,
        confidence=max_prob
    )
    db.add(db_record)
    db.commit()
    
    # Response
    probs = {name: float(prob) for name, prob in zip(models["target_names"], prediction_proba)}
    return {"species": predicted_species, "probabilities": probs}

@app.get("/history", response_model=list[HistoryOut])
def read_history(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    records = db.query(PredictionHistory).order_by(PredictionHistory.timestamp.desc()).offset(skip).limit(limit).all()
    return records
