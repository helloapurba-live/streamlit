import reflex as rx

def index() -> rx.Component:
    return rx.vstack(
        rx.heading("🌸 Iris Dataset Explorer & Classifier", size="8"),
        rx.divider(),
        rx.text(
            "Welcome to the Reflex version of the Iris Data Suite. This application provides real-time "
            "machine learning predictions and advanced data visualizations.",
            size="4",
        ),
        rx.card(
            rx.vstack(
                rx.heading("Overview", size="4"),
                rx.text(
                    "The Iris dataset contains 150 records of three iris species: Setosa, Versicolor, and Virginica. "
                    "In this app, you can explore the features, perform clustering, and run predictions using a Random Forest model.",
                ),
                align="start",
            ),
            width="100%",
        ),
        rx.grid(
            rx.card(
                rx.vstack(
                    rx.icon("layout", size=24),
                    rx.heading("Multiple Frameworks", size="3"),
                    rx.text("Switch between Streamlit, Dash, and Reflex versions."),
                )
            ),
            rx.card(
                rx.vstack(
                    rx.icon("database", size=24),
                    rx.heading("Shared Backend", size="3"),
                    rx.text("Same ML models and SQLite history database."),
                )
            ),
            rx.card(
                rx.vstack(
                    rx.icon("cpu", size=24),
                    rx.heading("Real-time ML", size="3"),
                    rx.text("Instant predictions based on your input dimensions."),
                )
            ),
            columns="3",
            spacing="4",
            width="100%",
        ),
        spacing="6",
    )
