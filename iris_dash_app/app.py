import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB])

sidebar = html.Div(
    [
        html.H2("Iris App", className="display-4"),
        html.Hr(),
        html.P(
            "Enterprise Dash Suite", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.I(className="bi bi-house-door me-2"), 
                        f"{page['name']}"
                    ], 
                    href=page["relative_path"],
                    active="exact"
                )
                for page in dash.page_registry.values()
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "18rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    },
)

content = html.Div(
    dash.page_container,
    style={
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    },
)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
