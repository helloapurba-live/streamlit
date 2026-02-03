import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.datasets import load_iris
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import streamlit.components.v1 as components

st.set_page_config(page_title="Model Monitoring", page_icon="🔍", layout="wide")

st.title("🔍 MLOps: Model Monitoring")
st.markdown("""
Using **EvidentlyAI**, we monitor **Data Drift**. 
We compare the **Reference Data** (original training set) vs **Current Production Data** (what users are inputting).
If the distributions diverge significantly, the model might need retraining.
""")

# Load Reference Data
@st.cache_data
def load_reference_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target # We treat species as roughly synonymous with target
    return df

reference_df = load_reference_data()

# Load Current Data from DB
DATABASE_URL = "sqlite:///./iris.db"
engine = create_engine(DATABASE_URL)

try:
    current_df_raw = pd.read_sql("SELECT * FROM history", engine)
    
    if len(current_df_raw) < 5:
        st.warning(f"Not enough production data to calculate drift (Found {len(current_df_raw)} records). Please make at least 5 predictions first.")
    else:
        # Align columns
        # DB has sepal_length, reference has 'sepal length (cm)'
        # We need to map them manually
        current_df = pd.DataFrame()
        current_df['sepal length (cm)'] = current_df_raw['sepal_length']
        current_df['sepal width (cm)'] = current_df_raw['sepal_width']
        current_df['petal length (cm)'] = current_df_raw['petal_length']
        current_df['petal width (cm)'] = current_df_raw['petal_width']
        
        # Generate Report
        report = Report(metrics=[
            DataDriftPreset(), 
        ])
        
        with st.spinner("Calculating Drift..."):
            report.run(reference_data=reference_df, current_data=current_df)
            
            # Save report to html
            report.save_html("drift_report.html")
            
            # Read back and display
            with open("drift_report.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            
            components.html(html_content, height=1000, scrolling=True)
            
except Exception as e:
    st.error(f"Error accessing database or calculating drift: {e}")
