import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from utils import load_data

dash.register_page(__name__, name='📊 General EDA', order=1)

df = load_data()
features = df.columns[:-1]

layout = dbc.Container([
    html.H2("📊 Exploratory Data Analysis"),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Feature for Analysis:"),
            dcc.Dropdown(
                id='eda-feature-dropdown',
                options=features,
                value=features[0],
                clearable=False
            )
        ], width=4)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Feature Distribution"),
                dbc.CardBody(dcc.Graph(id='eda-hist'))
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Feature Spread (Boxplot)"),
                dbc.CardBody(dcc.Graph(id='eda-box'))
            ])
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Bivariate Scatter Plot"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id='eda-scatter-x', options=features, value=features[0]), width=6),
                        dbc.Col(dcc.Dropdown(id='eda-scatter-y', options=features, value=features[1]), width=6)
                    ], className="mb-2"),
                    dcc.Graph(id='eda-scatter')
                ])
            ], className="mt-4")
        ])
    ])
])

@callback(
    Output('eda-hist', 'figure'),
    Output('eda-box', 'figure'),
    Input('eda-feature-dropdown', 'value')
)
def update_univariate(feature):
    fig_hist = px.histogram(df, x=feature, color="species", nbins=30, opacity=0.7, barmode='overlay')
    fig_hist.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    
    fig_box = px.box(df, x="species", y=feature, color="species", points="all")
    fig_box.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    
    return fig_hist, fig_box

@callback(
    Output('eda-scatter', 'figure'),
    Input('eda-scatter-x', 'value'),
    Input('eda-scatter-y', 'value')
)
def update_scatter(x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col, color="species", size='sepal width (cm)', hover_data=df.columns)
    fig.update_layout(height=500)
    return fig
