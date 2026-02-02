from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Global variables to store the model
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
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
    # Clean up input resources
    models.clear()

app = FastAPI(
    title="Iris Classification API",
    description="A simple API to predict Iris flower species.",
    version="1.0.0",
    lifespan=lifespan
)

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

@app.get("/")
def home():
    return {"message": "Welcome to the Iris API. Go to /docs for Swagger UI."}

@app.post("/predict", response_model=PredictionOut)
def predict(data: IrisInput):
    if "clf" not in models:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    # Convert input to dataframe/array
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    
    # Predict
    prediction_idx = models["clf"].predict(input_data)[0]
    prediction_proba = models["clf"].predict_proba(input_data)[0]
    
    predicted_species = models["target_names"][prediction_idx]
    
    # Map probabilities to class names
    probs = {
        name: float(prob) 
        for name, prob in zip(models["target_names"], prediction_proba)
    }
    
    return {
        "species": predicted_species,
        "probabilities": probs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
