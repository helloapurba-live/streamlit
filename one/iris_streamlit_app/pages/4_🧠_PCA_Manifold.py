import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PCA & Manifold", page_icon="🧠", layout="wide")

st.title("🧠 PCA & Dimensionality Reduction")
st.markdown("Principal Component Analysis (PCA) reduces the 4-dimensional Iris data into 2 or 3 dimensions to visualize the inherent structure.")

@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    return X, y, target_names

X, y, target_names = load_data()

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca_2d = PCA(n_components=2)
principal_components_2d = pca_2d.fit_transform(X_scaled)
pca_df_2d = pd.DataFrame(data=principal_components_2d, columns=['PC1', 'PC2'])
pca_df_2d['species'] = [target_names[i] for i in y]

pca_3d = PCA(n_components=3)
principal_components_3d = pca_3d.fit_transform(X_scaled)
pca_df_3d = pd.DataFrame(data=principal_components_3d, columns=['PC1', 'PC2', 'PC3'])
pca_df_3d['species'] = [target_names[i] for i in y]

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("2D PCA Projection")
    fig_2d = px.scatter(pca_df_2d, x='PC1', y='PC2', color='species', opacity=0.7)
    st.plotly_chart(fig_2d, use_container_width=True)
    
    st.info(f"Explained Variance (2D): {sum(pca_2d.explained_variance_ratio_):.2%}")

with col2:
    st.subheader("3D PCA Projection")
    fig_3d = px.scatter_3d(pca_df_3d, x='PC1', y='PC2', z='PC3', color='species', opacity=0.7)
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.info(f"Explained Variance (3D): {sum(pca_3d.explained_variance_ratio_):.2%}")

# Component Loadings
st.subheader("Feature Importance (Loadings)")
loadings = pd.DataFrame(pca_2d.components_.T, columns=['PC1', 'PC2'], index=load_iris().feature_names)
st.write("How much each original feature contributes to the principal components:")
st.dataframe(loadings.style.background_gradient(cmap="coolwarm"))
