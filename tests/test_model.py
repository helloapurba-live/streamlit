import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def test_model_training():
    """Test that the model trains and predicts correctly."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    # Test prediction shape
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = clf.predict(sample)
    
    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2] # Must be a valid class index

def test_data_integrity():
    """Test that dataset is consistent."""
    iris = load_iris()
    assert len(iris.data) == 150
    assert len(iris.feature_names) == 4
