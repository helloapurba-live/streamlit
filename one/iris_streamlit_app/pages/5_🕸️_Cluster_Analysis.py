import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

st.set_page_config(page_title="Cluster Analysis", page_icon="🕸️", layout="wide")

st.title("🕸️ Unsupervised Cluster Analysis")
st.markdown("Can the computer find the flower species **without** being told the answers? We use K-Means clustering to find groups.")

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Actual Species'] = [iris.target_names[i] for i in iris.target]
    return df, iris.target_names

df, target_names = load_data()
X = df.iloc[:, :-1] # Features only

# Interactive K
k_clusters = st.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=3)

# Run KMeans
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
df['Cluster Label'] = kmeans.fit_predict(X)
df['Cluster Label'] = df['Cluster Label'].astype(str) # For discrete coloring

# Side-by-Side Comparison
st.subheader(f"Comparision: Actual Species vs. K-Means (K={k_clusters})")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**1. Actual Ground Truth**")
    fig_actual = px.scatter(df, x=df.columns[2], y=df.columns[3], color='Actual Species', 
                            title="Actual Species Labels",
                            hover_data=['Actual Species'])
    st.plotly_chart(fig_actual, use_container_width=True)

with c2:
    st.markdown(f"**2. K-Means Clusters**")
    fig_cluster = px.scatter(df, x=df.columns[2], y=df.columns[3], color='Cluster Label', 
                             title=f"Discovered Clusters (K={k_clusters})",
                             color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_cluster, use_container_width=True)

st.warning("""
**Note**: The cluster colors and label IDs (0, 1, 2) are arbitrary and may not match the specific color/name of the actual species. 
Look for *grouping patterns* rather than matching colors.
""")
