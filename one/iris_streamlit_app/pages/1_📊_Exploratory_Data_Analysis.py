import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris EDA", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis")
st.markdown("Interact with the charts to explore patterns in the Iris dataset.")

# Load Data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in iris.target]
    return df

df = load_data()

# ---------------------------------------------------------
# 1. Dataset Overview
# ---------------------------------------------------------
st.subheader("1. Dataset Overview")
with st.expander("Results"):
    st.dataframe(df)
    st.write(f"Shape of the dataset: {df.shape}")
    st.write(df.describe())

# ---------------------------------------------------------
# 2. Univariate Analysis
# ---------------------------------------------------------
st.subheader("2. Univariate Analysis")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Feature Distribution (Histogram)**")
    selected_feature = st.selectbox("Select Feature for Histogram", df.columns[:-1])
    fig_hist = px.histogram(df, x=selected_feature, color="species", nbins=30, opacity=0.7, barmode='overlay')
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown("**Feature Spread (Box Plot)**")
    fig_box = px.box(df, x="species", y=selected_feature, color="species", points="all")
    st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------------------
# 3. Bivariate Analysis
# ---------------------------------------------------------
st.subheader("3. Bivariate Analysis")

c1, c2, c3 = st.columns([1, 1, 3])
with c1:
    x_axis = st.selectbox("X-Axis", df.columns[:-1], index=2)
with c2:
    y_axis = st.selectbox("Y-Axis", df.columns[:-1], index=3)

fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color="species", 
                         size='sepal width (cm)', hover_data=df.columns)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------
# 4. 3D Visualization
# ---------------------------------------------------------
st.subheader("4. 3D Multivariate Analysis")
st.markdown("Rotate the plot to see the separation between clusters.")

fig_3d = px.scatter_3d(df, x='sepal length (cm)', y='sepal width (cm)', z='petal length (cm)',
              color='species', symbol='species', opacity=0.7)
fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
st.plotly_chart(fig_3d, use_container_width=True)
