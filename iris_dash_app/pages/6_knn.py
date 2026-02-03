import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from utils import load_data
import numpy as np

dash.register_page(__name__, name='🔗 KNN Network', order=6)

df = load_data()
features = df.columns[:-1]
X = df[features].values

layout = dbc.Container([
    html.H2("🔗 K-Nearest Neighbors Network"),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select K (Neighbors):"),
            dcc.Slider(
                id='knn-k', min=2, max=20, step=1, value=5,
                marks={i: str(i) for i in range(2, 21, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=12)
    ], className="mb-4"),
    
    dcc.Graph(id='knn-graph', style={'height': '80vh'})
])

@callback(
    Output('knn-graph', 'figure'),
    Input('knn-k', 'value')
)
def update_knn(k):
    # Construct Graph
    A = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    G = nx.from_scipy_sparse_array(A)
    
    # Layout using Spring (3D)
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Extract Nodes
    x_nodes = [pos[i][0] for i in G.nodes()]
    y_nodes = [pos[i][1] for i in G.nodes()]
    z_nodes = [pos[i][2] for i in G.nodes()]
    
    # Colors for nodes
    species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    colors = [species_map[s] for s in df['species']]
    
    trace_nodes = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(symbol='circle', size=5, color=colors, colorscale='Viridis', line=dict(color='black', width=0.5)),
        text=df['species'],
        hoverinfo='text'
    )
    
    # Extract Edges
    x_edges = []
    y_edges = []
    z_edges = []
    
    for edge in G.edges():
        x_edges += [pos[edge[0]][0], pos[edge[1]][0], None]
        y_edges += [pos[edge[0]][1], pos[edge[1]][1], None]
        z_edges += [pos[edge[0]][2], pos[edge[1]][2], None]
        
    trace_edges = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='gray', width=1),
        hoverinfo='none'
    )
    
    layout = go.Layout(
        title=f"KNN Network Topology (K={k})",
        showlegend=False,
        scene=dict(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), zaxis=dict(showgrid=False))
    )
    
    return go.Figure(data=[trace_edges, trace_nodes], layout=layout)
