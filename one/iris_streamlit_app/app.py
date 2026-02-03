import streamlit as st

st.set_page_config(
    page_title="Iris ML App",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Dataset Explorer & Classifier")

st.markdown("""
### Welcome to the Interactive Iris App

This application demonstrates the power of Streamlit for Data Science. It consists of two main modules:

1.  **📊 Exploratory Data Analysis (EDA)**: Deep dive into the dataset with interactive Plotly visualizations.
2.  **🤖 Model Prediction**: A real-time machine learning model to classify iris flowers based on your inputs.

#### About the Dataset AND Model
The **Iris flower dataset** is a multivariate data set. It consists of 50 samples from each of three species of Iris (*Iris setosa*, *Iris virginica*, and *Iris versicolor*). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

---
👈 **Select a page from the sidebar to get started!**
""")

st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Versicolor")
