import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from sklearn.decomposition import PCA
from utils import load_data

dash.register_page(__name__, name='🧠 PCA Manifold', order=4)

df = load_data()
features = df.columns[:-1]

layout = dbc.Container([
    html.H2("🧠 PCA Analysis (3D Manifold)"),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.P("Projecting 4D Iris data into 3D using Principal Component Analysis."),
            dcc.Dropdown(
                id='pca-color',
                options=[{'label': 'Species', 'value': 'species'}] + [{'label': f, 'value': f} for f in features],
                value='species',
                clearable=False,
                className="mb-3"
            )
        ], width=4)
    ]),
    
    dcc.Graph(id='pca-graph', style={'height': '80vh'})
])

@callback(
    Output('pca-graph', 'figure'),
    Input('pca-color', 'value')
)
def update_pca(color_col):
    pca = PCA(n_components=3)
    components = pca.fit_transform(df[features])
    
    total_var = pca.explained_variance_ratio_.sum() * 100
    
    fig = px.scatter_3d(
        components, x=0, y=1, z=2,
        color=df[color_col],
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
        title=f"Total Explained Variance: {total_var:.2f}%",
        opacity=0.8
    )
    return fig
