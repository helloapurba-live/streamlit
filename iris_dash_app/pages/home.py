import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Home', order=0)

layout = html.Div([
    html.H1("🌸 Iris Dataset Explorer & Classifier"),
    html.Hr(),
    
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Welcome to the Dash Enterprise Suite"),
                dcc.Markdown("""
                This application demonstrates the power of **Dash Plotly** for Data Science. 
                It replicates the functionality of our Streamlit app with full feature parity:
                
                1.  **📊 EDA**: Interactive Plotly visualizations.
                2.  **🤖 Prediction**: Real-time inference.
                3.  **🕸️ Networks**: Graph theory analysis.
                4.  **💬 Chat**: AI-powered data analysis.
                
                ---
                **About the Dataset**:
                The Iris flower dataset is a multivariate data set. It consists of 50 samples from each of three species of Iris (*Iris setosa*, *Iris virginica*, and *Iris versicolor*).
                """),
                
                dbc.Alert(
                    "👈 Select a page from the sidebar to get started!", 
                    color="primary",
                    className="mt-4"
                )
            ], width=8),
            
            dbc.Col([
                html.Img(src="https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", 
                         className="img-fluid rounded shadow")
            ], width=4)
        ])
    ])
])
