import reflex as rx
import plotly.express as px
from ..utils import load_data

def stats() -> rx.Component:
    df = load_data()
    
    # Violin Plot
    fig_violin = px.violin(
        df, y="sepal length (cm)", x="species", 
        color="species", box=True, points="all",
        title="Sepal Length Distribution by Species"
    )
    
    # Parallel Coordinates
    fig_par = px.parallel_coordinates(
        df, color="sepal length (cm)",
        labels={col: col.replace(" (cm)", "") for col in df.columns},
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Multivariate Parallel Coordinates"
    )

    return rx.vstack(
        rx.heading("📉 Statistical Insights", size="8"),
        rx.divider(),
        rx.text("Deep dive into the multivariate relationships within the Iris dataset."),
        rx.grid(
            rx.card(rx.plotly(data=fig_violin, height="400px")),
            rx.card(rx.plotly(data=fig_par, height="400px")),
            columns="1",
            spacing="6",
            width="100%",
        ),
        spacing="6",
    )
