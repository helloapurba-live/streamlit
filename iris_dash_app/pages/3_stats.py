import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from utils import load_data

dash.register_page(__name__, name='📉 Stats Insights', order=3)

df = load_data()
features = df.columns[:-1]

# Factorize for parcoords
df['species_id'] = pd.factorize(df['species'])[0]

layout = dbc.Container([
    html.H2("📉 Statistical Insights"),
    html.Hr(),
    
    html.H4("1. Feature Density (Violin Plot)"),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='stats-violin-feature', options=features, value=features[0], clearable=False),
            dcc.Graph(id='stats-violin')
        ], width=12)
    ], className="mb-5"),
    
    html.H4("2. Multivariate Parallel Coordinates"),
    dcc.Graph(
        figure=px.parallel_coordinates(
            df, 
            color="species_id",
            labels={"species_id": "Species"},
            color_continuous_scale=px.colors.diverging.Tealrose,
            color_continuous_midpoint=1
        )
    ),
    
    html.H4("3. Correlation Heatmap", className="mt-5"),
    dcc.Graph(
        figure=px.imshow(
            df.drop(['species', 'species_id'], axis=1).corr(), 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale="Viridis"
        )
    )
])

@callback(
    Output('stats-violin', 'figure'),
    Input('stats-violin-feature', 'value')
)
def update_violin(feature):
    return px.violin(df, y=feature, x="species", color="species", box=True, points="all")
