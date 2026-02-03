import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from utils import load_data
import os

dash.register_page(__name__, name='💬 Chat with Data', order=9)

df = load_data()

layout = dbc.Container([
    html.H2("💬 Chat with your Data"),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            dbc.Input(id='openai-key', type='password', placeholder="Enter OpenAI API Key", className="mb-3"),
            dbc.Textarea(id='chat-input', placeholder="Ask a question about the Iris dataset...", style={'height': 100}, className="mb-3"),
            dbc.Button("Send", id='chat-send', color="primary", className="mb-4"),
            
            html.Div(id='chat-response')
        ], width=8),
        
        dbc.Col([
            html.H5("Sample Questions"),
            html.Ul([
                html.Li("What is the average sepal length?"),
                html.Li("Plot a bar chart of species counts."),
                html.Li("Which flower has the widest petal?")
            ])
        ], width=4)
    ])
])

@callback(
    Output('chat-response', 'children'),
    Input('chat-send', 'n_clicks'),
    State('chat-input', 'value'),
    State('openai-key', 'value'),
    prevent_initial_call=True
)
def process_chat(n, prompt, api_key):
    if not prompt or not api_key:
        return dbc.Alert("Please enter both an API Key and a Question.", color="warning")
    
    try:
        llm = OpenAI(api_token=api_key)
        smart_df = SmartDataframe(df, config={"llm": llm})
        
        response = smart_df.chat(prompt)
        
        if str(response).endswith(".png") and os.path.exists(str(response)):
            import base64
            encoded_image = base64.b64encode(open(str(response), 'rb').read()).decode('ascii')
            return html.Img(src='data:image/png;base64,{}'.format(encoded_image), className="img-fluid")
        else:
            return dbc.Card(dbc.CardBody(str(response)))
            
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger")
