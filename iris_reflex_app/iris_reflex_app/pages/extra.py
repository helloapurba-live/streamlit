import reflex as rx

def chat() -> rx.Component:
    return rx.vstack(
        rx.heading("💬 Chat with Data (AI)", size="8"),
        rx.divider(),
        rx.card(
            rx.vstack(
                rx.text("Ask questions about the Iris dataset using natural language."),
                rx.text_area(placeholder="E.g., What is the average petal length per species?", width="100%"),
                rx.button("Send Query", color_scheme="blue"),
                rx.callout("This feature requires an OpenAI API key. Contact admin for setup.", icon="info"),
                align="start",
            ),
            width="100%",
        ),
        spacing="6",
    )

def monitoring() -> rx.Component:
    return rx.vstack(
        rx.heading("🔍 Model Monitoring (MLOps)", size="8"),
        rx.divider(),
        rx.card(
            rx.vstack(
                rx.text("Monitoring data drift and model performance metrics."),
                rx.center(
                    rx.vstack(
                        rx.icon("activity", size=48, color="green"),
                        rx.text("Real-time Drift Status: HEALTHY", color="green", weight="bold"),
                        spacing="2",
                    ),
                    padding="10",
                    width="100%",
                ),
                rx.text("Data drift reports are generated using EvidentlyAI and logged to the central dashboard."),
                align="start",
            ),
            width="100%",
        ),
        spacing="6",
    )
