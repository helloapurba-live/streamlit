import reflex as rx
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import io
import base64
from ..utils import get_model_and_data

def tree_page() -> rx.Component:
    model, _, feature_names, _ = get_model_and_data()
    
    # Generate Tree plot
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    img_str = base64.b64encode(buf.getvalue()).decode()
    
    return rx.vstack(
        rx.heading("🌲 Decision Tree Logical Flow", size="8"),
        rx.divider(),
        rx.text("Visualizing the decision-making process of the Random Forest model."),
        rx.card(
            rx.image(src=f"data:image/png;base64,{img_str}", width="100%"),
            width="100%",
        ),
        spacing="6",
    )
