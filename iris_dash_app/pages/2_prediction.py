import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from utils import get_model_and_data

dash.register_page(__name__, name='🤖 Prediction', order=2)

clf, target_names, feature_names, X = get_model_and_data()

# Calculate ranges for sliders
ranges = {}
for i, f in enumerate(feature_names):
    ranges[f] = {
        'min': float(X[:, i].min()),
        'max': float(X[:, i].max()),
        'mean': float(X[:, i].mean())
    }

layout = dbc.Container([
    html.H2("🤖 Real-time Species Predictor"),
    html.Hr(),
    
    dbc.Row([
        # Left Column: Inputs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Input Measurements"),
                dbc.CardBody([
                    html.Label(f"{feature_names[0]} (cm)"),
                    dcc.Slider(id='sl-sepal-l', min=ranges[feature_names[0]]['min'], max=ranges[feature_names[0]]['max'], value=ranges[feature_names[0]]['mean'], tooltip={"placement": "bottom", "always_visible": True}),
                    
                    html.Label(f"{feature_names[1]} (cm)", className="mt-3"),
                    dcc.Slider(id='sl-sepal-w', min=ranges[feature_names[1]]['min'], max=ranges[feature_names[1]]['max'], value=ranges[feature_names[1]]['mean'], tooltip={"placement": "bottom", "always_visible": True}),
                    
                    html.Label(f"{feature_names[2]} (cm)", className="mt-3"),
                    dcc.Slider(id='sl-petal-l', min=ranges[feature_names[2]]['min'], max=ranges[feature_names[2]]['max'], value=ranges[feature_names[2]]['mean'], tooltip={"placement": "bottom", "always_visible": True}),
                    
                    html.Label(f"{feature_names[3]} (cm)", className="mt-3"),
                    dcc.Slider(id='sl-petal-w', min=ranges[feature_names[3]]['min'], max=ranges[feature_names[3]]['max'], value=ranges[feature_names[3]]['mean'], tooltip={"placement": "bottom", "always_visible": True}),
                ])
            ])
        ], width=4),
        
        # Right Column: Result + Visualization
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div(id='prediction-output', className="text-center mb-4")
                ])
            ]),
            dbc.Row([
               dcc.Graph(id='prediction-viz') 
            ])
        ], width=8)
    ])
])

@callback(
    Output('prediction-output', 'children'),
    Output('prediction-viz', 'figure'),
    Input('sl-sepal-l', 'value'),
    Input('sl-sepal-w', 'value'),
    Input('sl-petal-l', 'value'),
    Input('sl-petal-w', 'value')
)
def make_prediction(sl, sw, pl, pw):
    input_df = [[sl, sw, pl, pw]]
    pred_idx = clf.predict(input_df)[0]
    pred_proba = clf.predict_proba(input_df)[0]
    species = target_names[pred_idx]
    
    # Create Result Card
    color_map = {'setosa': 'success', 'versicolor': 'warning', 'virginica': 'danger'}
    color = color_map.get(species, 'light')
    
    result_card = dbc.Alert([
        html.H3(species.capitalize(), className="display-3"),
        html.P(f"Confidence: {max(pred_proba):.2%}")
    ], color=color)
    
    # Create Visualization (Scatter context)
    df_full = pd.DataFrame(X, columns=feature_names)
    df_full['species'] = [target_names[i] for i in clf.predict(X)] # Using model preds for consistency/lazy loading
    
    fig = px.scatter(df_full, x=feature_names[2], y=feature_names[3], color='species', opacity=0.5)
    # Add user point
    fig.add_scatter(x=[pl], y=[pw], mode='markers', marker=dict(color='red', size=20, symbol='star'), name='Your Input')
    fig.update_layout(title="Your Input vs. Dataset (Petal Dimensions)")
    
    return result_card, fig
