import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import networkx as nx
from utils import load_data

dash.register_page(__name__, name='🕸️ Feature Network', order=7)

df = load_data()

layout = dbc.Container([
    html.H2("🕸️ Feature Correlation Network"),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Correlation Threshold:"),
            dcc.Slider(
                id='feat-threshold', min=0, max=1, step=0.1, value=0.5,
                marks={i/10: str(i/10) for i in range(0, 11)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=12)
    ], className="mb-4"),
    
    dcc.Graph(id='feat-graph', style={'height': '70vh'})
])

@callback(
    Output('feat-graph', 'figure'),
    Input('feat-threshold', 'value')
)
def update_feat_network(threshold):
    # Calculate Correlation
    corr_matrix = df.drop('species', axis=1).corr().abs()
    
    # Build Graph
    G = nx.Graph()
    for col in corr_matrix.columns:
        G.add_node(col)
        
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr_matrix.iloc[i, j])
    
    pos = nx.spring_layout(G, seed=42)
    
    # Edges
    edge_x = []
    edge_y = []
    weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weights.append(f"{edge[2]['weight']:.2f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Nodes
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=30,
            color=[len(list(G.neighbors(n))) for n in G.nodes()],
        ))
        
    layout = go.Layout(
        title=f"Feature Relationships (Corr > {threshold})",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return go.Figure(data=[edge_trace, node_trace], layout=layout)
