import reflex as rx
import plotly.express as px
from ..utils import load_data
from ..state import State

def eda() -> rx.Component:
    df = load_data()
    fig = px.histogram(
        df, 
        x="sepal length (cm)", 
        color="species", 
        marginal="box",
        title="Feature Distribution by Species"
    )
    
    return rx.vstack(
        rx.heading("📊 General EDA", size="8"),
        rx.divider(),
        rx.text("Explore the distributions of Iris features across species."),
        rx.card(
            rx.plotly(data=fig, height="500px"),
            width="100%",
        ),
        rx.grid(
            rx.card(
                rx.vstack(
                    rx.heading("Key Insight", size="3"),
                    rx.text("Setosa is highly separable based on petal length and width."),
                )
            ),
            rx.card(
                rx.vstack(
                    rx.heading("Correlation", size="3"),
                    rx.text("Strong positive correlation between petal dimensions."),
                )
            ),
            columns="2",
            spacing="4",
            width="100%",
        ),
        spacing="6",
    )
