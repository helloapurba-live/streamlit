import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Feature Network", page_icon="🕸️", layout="wide")

st.title("🕸️ Feature Interaction Network")
st.markdown("Visualizing the relationships **between variables** rather than samples. Stronger correlations create thicker connections.")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Calculate Feature Importance for Node Size
model = RandomForestClassifier()
model.fit(iris.data, iris.target)
importance = model.feature_importances_
node_sizes = [10 + (imp * 100) for imp in importance] # Scale for visual

# Calculate Correlation for Edges
corr_matrix = df.corr().abs()

# User Threshold
threshold = st.slider("Min Correlation Threshold to Show Connection", 0.0, 1.0, 0.5)

# Build Graph
G = nx.Graph()
cols = df.columns
for col in cols:
    G.add_node(col)

for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        val = corr_matrix.iloc[i, j]
        if val >= threshold:
            G.add_edge(cols[i], cols[j], weight=val)

# Spring Layout
pos = nx.spring_layout(G, seed=42)

# Edges Trace
edge_x = []
edge_y = []
weights = []

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    weights.append(edge[2]['weight'])

# Dynamic edge width based on weight
# In Plotly scatter mode='lines', we can't vary width easily per segment in one trace.
# We will use opacity and annotation to imply strength, or just a uniform width for simplicity in 2D.
# For a robust "web", we'll just draw them. To do variable width, we'd need multiple traces or shapes.
# We'll use a loop for individual edge traces to show thickness.
    
fig = go.Figure()

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    weight = edge[2]['weight']
    
    fig.add_trace(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        mode='lines',
        line=dict(width=weight*10, color='#888'), # Thickness based on correlation
        hoverinfo='none'
    ))

# Nodes Trace
node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=list(G.nodes()),
    textposition="top center",
    marker=dict(
        showscale=True,
        colorscale='Viridis',
        size=node_sizes,
        color=importance,
        colorbar=dict(title='RF Importance')
    )
)

fig.add_trace(node_trace)
fig.update_layout(title="Feature Correlation Structure", showlegend=False)
st.plotly_chart(fig, use_container_width=True)
