import reflex as rx
from .utils import get_model_and_data, load_data
import pandas as pd
import numpy as np

# Load model once at module level
model, target_names, feature_names, X_raw = get_model_and_data()

class State(rx.State):
    """The app state."""
    
    # Prediction inputs
    sepal_length: float = 5.1
    sepal_width: float = 3.5
    petal_length: float = 1.4
    petal_width: float = 0.2
    
    # Prediction output
    predicted_species: str = "setosa"
    prediction_confidence: float = 0.0
    
    # Sidebar state
    sidebar_open: bool = True
    
    def toggle_sidebar(self):
        self.sidebar_open = not self.sidebar_open
        
    @rx.var
    def prediction_result_card_color(self) -> str:
        colors = {"setosa": "green", "versicolor": "orange", "virginica": "red"}
        return colors.get(self.predicted_species, "gray")

    def run_prediction(self, val):
        # We handle slider changes here
        input_data = np.array([[
            self.sepal_length, 
            self.sepal_width, 
            self.petal_length, 
            self.petal_width
        ]])
        
        prob = model.predict_proba(input_data)[0]
        # Find index of max prob
        idx = np.argmax(prob)
        self.predicted_species = target_names[idx]
        self.prediction_confidence = float(np.max(prob))

    # Helper for layout
    def set_val(self, field: str, value: float):
        setattr(self, field, float(value))
        self.run_prediction(None)
