import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from sqlalchemy import create_engine
from sklearn.datasets import load_iris
import os

# Try importing evidently, handle failure gracefully
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
    EVIDENTLY_ERROR = None
except Exception as e:
    EVIDENTLY_AVAILABLE = False
    EVIDENTLY_ERROR = str(e)

dash.register_page(__name__, name='🔍 Model Monitoring', order=11)

layout = dbc.Container([
    html.H2("🔍 MLOps: Model Monitoring"),
    html.Hr(),
    html.Div(id='monitor-report')
])

def generate_report():
    if not EVIDENTLY_AVAILABLE:
        return dbc.Alert(
            [
                html.H4("EvidentlyAI Library Unavailable", className="alert-heading"),
                html.P("The monitoring library could not be loaded due to an environment conflict (Pydantic version mismatch)."),
                html.Hr(),
                html.P(f"Error Details: {EVIDENTLY_ERROR}", className="mb-0")
            ], 
            color="danger"
        )

    try:
        # Load Ref
        iris = load_iris()
        ref_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        ref_df['target'] = iris.target
        
        # Load Current
        # Adjust path for consistency depending on where app is run
        db_paths = ["sqlite:///../iris_streamlit_app/iris.db", "sqlite:///iris.db"]
        engine = None
        for path in db_paths:
            try:
                eng = create_engine(path)
                with eng.connect() as conn:
                    pass
                engine = eng
                break
            except:
                continue
        
        if not engine:
             return html.Div("Could not connect to database (iris.db not found).")
            
        try:
            curr_df_raw = pd.read_sql("SELECT * FROM history", engine)
        except:
             return html.Div("Database found but no history table.")

        if len(curr_df_raw) < 5:
            return dbc.Alert("Not enough data to calculate drift. (Need at least 5 predictions)", color="warning")
            
        # Map columns
        curr_df = pd.DataFrame()
        curr_df['sepal length (cm)'] = curr_df_raw['sepal_length']
        curr_df['sepal width (cm)'] = curr_df_raw['sepal_width']
        curr_df['petal length (cm)'] = curr_df_raw['petal_length']
        curr_df['petal width (cm)'] = curr_df_raw['petal_width']
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=curr_df)
        
        # Save to assets folder so Dash can serve it
        if not os.path.exists("assets"):
            os.makedirs("assets")
            
        report.save_html("assets/drift_report.html")
        
        return html.Iframe(src="/assets/drift_report.html", style={"width": "100%", "height": "1000px", "border": "none"})
        
    except Exception as e:
        return dbc.Alert(f"Error generating report: {e}", color="danger")

layout.children[2] = generate_report()
