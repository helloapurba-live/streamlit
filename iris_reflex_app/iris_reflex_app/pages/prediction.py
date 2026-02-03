import reflex as rx
from ..state import State

def prediction_slider(label: str, value_var: rx.Var, on_change_fn) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(label, weight="bold"),
            rx.spacer(),
            rx.text(value_var.to_string()),
            width="100%",
        ),
        rx.slider(
            value=value_var,
            on_change=on_change_fn,
            min=0.1, max=8.0, step=0.1,
        ),
        width="100%",
        spacing="1",
    )

def prediction() -> rx.Component:
    return rx.vstack(
        rx.heading("🤖 Real-time Prediction", size="8"),
        rx.divider(),
        rx.grid(
            rx.card(
                rx.vstack(
                    rx.heading("Input Dimensions", size="4"),
                    prediction_slider("Sepal Length", State.sepal_length, lambda v: State.set_val("sepal_length", v)),
                    prediction_slider("Sepal Width", State.sepal_width, lambda v: State.set_val("sepal_width", v)),
                    prediction_slider("Petal Length", State.petal_length, lambda v: State.set_val("petal_length", v)),
                    prediction_slider("Petal Width", State.petal_width, lambda v: State.set_val("petal_width", v)),
                    width="100%",
                    spacing="4",
                ),
                padding="6",
            ),
            rx.card(
                rx.vstack(
                    rx.heading("Prediction Result", size="4"),
                    rx.divider(),
                    rx.center(
                        rx.vstack(
                            rx.text("Predicted Species:", size="5"),
                            rx.heading(
                                State.predicted_species.to_string().upper(),
                                size="9",
                                color_scheme=State.prediction_result_card_color,
                            ),
                            rx.text(
                                f"Confidence: {(State.prediction_confidence * 100).to_string()}%",
                                size="4",
                            ),
                            spacing="4",
                        ),
                        height="100%",
                    ),
                    width="100%",
                    spacing="4",
                ),
                padding="6",
                background=rx.color(State.prediction_result_card_color, 2),
            ),
            columns="2",
            spacing="6",
            width="100%",
        ),
        spacing="6",
    )
