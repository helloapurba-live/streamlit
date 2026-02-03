import reflex as rx
import pandas as pd
from sqlalchemy import create_engine
import os

def history() -> rx.Component:
    # Try to find the shared database
    db_path = "../../iris_streamlit_app/iris.db"
    has_db = os.path.exists(db_path)
    
    rows = []
    if has_db:
        try:
            engine = create_engine(f"sqlite:///{db_path}")
            df_history = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10", engine)
            # Convert to list of dicts for Reflex table
            rows = df_history.values.tolist()
            cols = df_history.columns
        except Exception as e:
            error_msg = str(e)
            has_db = False
    
    return rx.vstack(
        rx.heading("📜 Prediction History", size="8"),
        rx.divider(),
        rx.cond(
            has_db,
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        *[rx.table.column_header_cell(col) for col in cols]
                    )
                ),
                rx.table.body(
                    *[
                        rx.table.row(
                            *[rx.table.cell(str(val)) for val in row]
                        )
                        for row in rows
                    ]
                ),
                width="100%",
            ),
            rx.callout(
                "Prediction database not found or empty. Make some predictions in the Streamlit or Dash apps first!",
                icon="alert_triangle",
                color_scheme="amber",
                width="100%",
            ),
        ),
        spacing="6",
    )
