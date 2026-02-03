import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from sklearn.cluster import KMeans
from utils import load_data

dash.register_page(__name__, name='🕸️ Cluster Analysis', order=5)

df = load_data()
features = df.columns[:-1]

layout = dbc.Container([
    html.H2("🕸️ K-Means Cluster Analysis"),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Number of Clusters (K):"),
            dcc.Slider(
                id='cluster-k', min=2, max=10, step=1, value=3,
                marks={i: str(i) for i in range(2, 11)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=6),
        
        dbc.Col([
            html.Label("Select X Axis:"),
            dcc.Dropdown(id='cluster-x', options=features, value=features[2], clearable=False),
        ], width=3),
        
        dbc.Col([
            html.Label("Select Y Axis:"),
            dcc.Dropdown(id='cluster-y', options=features, value=features[3], clearable=False),
        ], width=3),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='cluster-graph'), width=12)
    ])
])

@callback(
    Output('cluster-graph', 'figure'),
    Input('cluster-k', 'value'),
    Input('cluster-x', 'value'),
    Input('cluster-y', 'value')
)
def update_cluster(k, x_col, y_col):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(df[features])
    
    df['cluster'] = model.labels_.astype(str)
    
    fig = px.scatter(
        df, x=x_col, y=y_col, 
        color='cluster', 
        symbol='species',
        title=f"K-Means Clustering (K={k})"
    )
    fig.add_scatter(
        x=model.cluster_centers_[:, list(features).index(x_col)],
        y=model.cluster_centers_[:, list(features).index(y_col)],
        mode='markers',
        marker=dict(color='black', size=15, symbol='x'),
        name='Centroids'
    )
    return fig
