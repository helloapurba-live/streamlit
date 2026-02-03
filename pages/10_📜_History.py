import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

st.set_page_config(page_title="Prediction History", page_icon="📜", layout="wide")

st.title("📜 Prediction History")
st.markdown("Live log of all requests sent to the API. This uses **SQLite** to persist data.")

# Database Connection
DATABASE_URL = "sqlite:///./iris.db"
engine = create_engine(DATABASE_URL)

try:
    # Read entire history
    query = "SELECT * FROM history ORDER BY timestamp DESC"
    df = pd.read_sql(query, engine)
    
    if df.empty:
        st.warning("No history found. Try making some predictions on the Prediction page or via API!")
    else:
        # Metrics
        st.subheader("Live Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Requests", len(df))
        col2.metric("Most Common Species", df['predicted_species'].mode()[0])
        col3.metric("Avg Confidence", f"{df['confidence'].mean():.2%}")

        # Data Table
        st.dataframe(df, use_container_width=True)
        
        # Trend Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Requests over Time")
            # Resample by minute/hour if we had more data, for now just a bar chart of count per species
            fig_bar = px.bar(df['predicted_species'].value_counts(), 
                             title="Prediction Counts by Species",
                             labels={'value': 'Count', 'index': 'Species'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            st.subheader("Input Values Distribution")
            fig_hist = px.histogram(df, x='petal_length', color='predicted_species', 
                                    title="User Inputted Petal Lengths", nbins=20)
            st.plotly_chart(fig_hist, use_container_width=True)

except Exception as e:
    st.error(f"Could not connect to database: {e}")
    st.info("Make sure you have run the API at least once to create the database file.")
