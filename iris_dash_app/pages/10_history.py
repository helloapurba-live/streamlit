import dash
from dash import html, dash_table, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

dash.register_page(__name__, name='📜 History', order=10)

# Connect to the same DB as the API
DATABASE_URL = "sqlite:///../iris_streamlit_app/iris.db" # Assuming relative path
# Fallback to local if running independently
LOCAL_DB = "sqlite:///iris.db"

layout = dbc.Container([
    html.H2("📜 Prediction History"),
    html.Hr(),
    
    dbc.Row([
        dbc.Col(html.Div(id='history-stats'), width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(html.Div(id='history-table'), width=12)
    ]),
    
    dcc.Interval(id='history-refresh', interval=5000, n_intervals=0) # Auto-refresh every 5s
])

@callback(
    Output('history-table', 'children'),
    Output('history-stats', 'children'),
    Input('history-refresh', 'n_intervals')
)
def update_history(n):
    try:
        engine = create_engine(DATABASE_URL)
        # Try connection
        with engine.connect() as conn:
            pass
    except:
        engine = create_engine(LOCAL_DB)
        
    try:
        df = pd.read_sql("SELECT * FROM history ORDER BY timestamp DESC", engine)
        if df.empty:
            return dbc.Alert("No history found.", color="info"), ""
            
        # Table
        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )
        
        # Stats
        stats = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Total Requests"), html.H2(len(df))])), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Most Common"), html.H2(df['predicted_species'].mode()[0])])), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Avg Confidence"), html.H2(f"{df['confidence'].mean():.2%}")])), width=4),
        ])
        
        return table, stats
        
    except Exception as e:
        return dbc.Alert(f"Database error: {e}", color="danger"), ""
