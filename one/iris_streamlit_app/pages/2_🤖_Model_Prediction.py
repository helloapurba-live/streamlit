import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Iris Prediction", page_icon="🤖", layout="wide")

# Custom CSS (Recycled for consistency)
st.markdown("""
<style>
    div.stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Species Predictor")

# Load and Train (Same logic as before)
@st.cache_data
def load_data_and_train():
    iris = load_iris()
    X = iris.data
    y = iris.target
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf, iris.target_names, iris.feature_names, iris.data, pd.DataFrame(iris.data, columns=iris.feature_names)

clf, target_names, feature_names, X, full_df = load_data_and_train()

# Sidebar
st.sidebar.header('Input Parameters')
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
    sepal_width = st.sidebar.slider('Sepal Width (cm)', float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
    petal_length = st.sidebar.slider('Petal Length (cm)', float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
    petal_width = st.sidebar.slider('Petal Width (cm)', float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))
    
    data = {feature_names[0]: sepal_length,
            feature_names[1]: sepal_width,
            feature_names[2]: petal_length,
            feature_names[3]: petal_width}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main Area
col1, col2 = st.columns([1, 2])

prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)
predicted_species = target_names[prediction][0]

with col1:
    st.subheader("Result")
    st.metric(label="Predicted Species", value=predicted_species.capitalize())
    
    st.write("Confidence Scores:")
    st.bar_chart(pd.DataFrame(prediction_proba, columns=target_names).T)

with col2:
    st.subheader("Visual Context")
    # Quick overlay using matplotlib again for specific single-point context
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=full_df, x=feature_names[2], y=feature_names[3], hue=load_iris().target, palette='viridis', alpha=0.5, ax=ax)
    # Re-map legend
    ax.legend(title='Species', labels=[n.capitalize() for n in target_names])
    
    ax.scatter(input_df[feature_names[2]], input_df[feature_names[3]], color='red', s=150, marker='*', label='You', zorder=5)
    plt.title("Your Input (Red Star) vs. Dataset")
    st.pyplot(fig)
