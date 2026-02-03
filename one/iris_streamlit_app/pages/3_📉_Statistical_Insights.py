import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

st.set_page_config(page_title="Statistical Insights", page_icon="📉", layout="wide")

st.title("📉 Statistical Insights")
st.markdown("Deep dive into the statistical distributions and multivariate relationships.")

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in iris.target]
    return df

df = load_data()

# 1. Violin Plots
st.subheader("1. Feature Density (Violin Plots)")
st.markdown("Violin plots show the probability density of the data at different values, combining a box plot with a kernel density plot.")

selected_feature = st.selectbox("Select Feature", df.columns[:-1])
fig_violin = px.violin(df, y=selected_feature, x="species", color="species", box=True, points="all", hover_data=df.columns)
st.plotly_chart(fig_violin, use_container_width=True)

# 2. Parallel Coordinates
st.subheader("2. Parallel Coordinates")
st.markdown("This plot maps each row in the data set as a line. It's powerful for seeing how classes separate across all dimensions simultaneously.")

# Factorize species for color mapping in numerical plot
df['species_id'] = pd.factorize(df['species'])[0]
fig_parcoords = px.parallel_coordinates(df, color="species_id", 
                                        labels={"species_id": "Species"},
                                        color_continuous_scale=px.colors.diverging.Tealrose,
                                        color_continuous_midpoint=1)
st.plotly_chart(fig_parcoords, use_container_width=True)

# 3. Correlation Heatmap
st.subheader("3. Feature Correlation")
corr = df.drop(['species', 'species_id'], axis=1).corr()
fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
st.plotly_chart(fig_heatmap, use_container_width=True)
