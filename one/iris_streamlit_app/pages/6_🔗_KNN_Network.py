import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA

st.set_page_config(page_title="KNN Network Topology", page_icon="🔗", layout="wide")

st.title("🔗 KNN Network Topology")
st.markdown("""
This visualization treats every single flower sample as a **Node** in a network. 
Edges are drawn between a flower and its nearest neighbors. This reveals the "shape" of the species clusters in 3D space.
""")

# Load Data
iris = load_iris()
X = iris.data
target = iris.target
target_names = iris.target_names

# User Controls
k_neighbors = st.slider("Number of Neighbors (K) to Connect", 2, 20, 5)

# 1. Build Graph
# Create adjacency matrix (connectivity)
A = kneighbors_graph(X, k_neighbors, mode='connectivity', include_self=False)
G = nx.from_scipy_sparse_array(A)

# 2. Get Layout using PCA (to keep the shape relevant to the data)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Assign 3D positions to nodes
pos = {i: X_pca[i] for i in range(len(X))}

# 3. Create Plotly Trace for Edges
edge_x = []
edge_y = []
edge_z = []

for edge in G.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])

edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# 4. Create Plotly Trace for Nodes
node_x = [pos[i][0] for i in G.nodes()]
node_y = [pos[i][1] for i in G.nodes()]
node_z = [pos[i][2] for i in G.nodes()]

node_color_map = {0: '#ffadad', 1: '#ffd6a5', 2: '#fdffb6'} # Pastel
node_colors = [node_color_map[target[i]] for i in G.nodes()]
node_text = [f"Flower #{i}<br>Species: {target_names[target[i]]}" for i in G.nodes()]

node_trace = go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    hoverinfo='text',
    text=node_text,
    marker=dict(
        showscale=False,
        color=node_colors,
        size=6,
        line_width=1
    )
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f'KNN Topology (3D PCA Layout) - {k_neighbors} Neighbors',
                    showlegend=False,
                    margin=dict(b=0,l=0,r=0,t=40),
                    scene=dict(
                        xaxis=dict(title='PC1'),
                        yaxis=dict(title='PC2'),
                        zaxis=dict(title='PC3')
                    )
                ))

st.plotly_chart(fig, use_container_width=True)
