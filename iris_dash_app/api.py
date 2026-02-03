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

app = FastAPI(title="Iris Classification API", version="2.0.0", lifespan=lifespan)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOut(BaseModel):
    species: str
    probabilities: dict[str, float]

@app.get("/")
def home():
    return {"message": "Welcome to the Iris API v2 (Dash Edition)."}

@app.post("/predict", response_model=PredictionOut)
def predict(data: IrisInput, db: Session = Depends(get_db)):
    if "clf" not in models:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction_idx = models["clf"].predict(input_data)[0]
    prediction_proba = models["clf"].predict_proba(input_data)[0]
    predicted_species = models["target_names"][prediction_idx]
    
    max_prob = float(max(prediction_proba))
    db_record = PredictionHistory(
        sepal_length=data.sepal_length, sepal_width=data.sepal_width,
        petal_length=data.petal_length, petal_width=data.petal_width,
        predicted_species=predicted_species, confidence=max_prob
    )
    db.add(db_record)
    db.commit()
    
    probs = {name: float(prob) for name, prob in zip(models["target_names"], prediction_proba)}
    return {"species": predicted_species, "probabilities": probs}

@app.get("/history")
def read_history(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(PredictionHistory).order_by(PredictionHistory.timestamp.desc()).offset(skip).limit(limit).all()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
