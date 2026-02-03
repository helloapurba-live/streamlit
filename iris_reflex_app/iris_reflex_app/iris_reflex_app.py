import reflex as rx
from .state import State
from .pages.index import index
from .pages.prediction import prediction
from .pages.eda import eda

def sidebar_item(text: str, icon: str, url: str) -> rx.Component:
    return rx.link(
        rx.hstack(
            rx.icon(icon, size=16),
            rx.text(text),
            spacing="3",
            padding_x="4",
            padding_y="2",
            border_radius="md",
            _hover={"bg": rx.color("accent", 3)},
        ),
        href=url,
        width="100%",
        text_decoration="none",
    )

def sidebar() -> rx.Component:
    return rx.vstack(
        rx.heading("Iris Reflex", size="5", margin_bottom="4"),
        sidebar_item("Home", "home", "/"),
        sidebar_item("General EDA", "bar-chart-2", "/eda"),
        sidebar_item("Prediction", "target", "/prediction"),
        sidebar_item("Stats Insights", "trending-down", "/stats"),
        sidebar_item("PCA Manifold", "brain", "/pca"),
        sidebar_item("Cluster Analysis", "webhook", "/cluster"),
        sidebar_item("Network Graph", "share-2", "/network"),
        sidebar_item("Decision Tree", "tree-pine", "/tree"),
        sidebar_item("History", "history", "/history"),
        sidebar_item("Chat", "message-square", "/chat"),
        sidebar_item("Monitoring", "search", "/monitoring"),
        rx.spacer(),
        rx.text("Version 1.0", size="1", color_alpha="0.5"),
        height="100vh",
        width="250px",
        padding="8",
        bg=rx.color("gray", 2),
        border_right=f"1px solid {rx.color('gray', 4)}",
        style={"position": "fixed", "left": 0, "top": 0},
    )

def theme_toggle() -> rx.Component:
    return rx.box(
        rx.color_mode.button(),
        position="fixed",
        top="4",
        right="4",
        z_index="1000",
    )

def layout(content: rx.Component) -> rx.Component:
    return rx.hstack(
        sidebar(),
        rx.box(
            theme_toggle(),
            content,
            margin_left="250px",
            padding="8",
            width="calc(100% - 250px)",
            min_height="100vh",
        ),
        spacing="0",
        width="100%",
    )

from .pages.stats import stats
from .pages.pca import pca_page
from .pages.cluster import cluster
from .pages.network import network
from .pages.history import history
from .pages.tree import tree_page
from .pages.extra import chat, monitoring

app = rx.App()
app.add_page(lambda: layout(index()), route="/", title="Iris - Home")
app.add_page(lambda: layout(eda()), route="/eda", title="Iris - EDA")
app.add_page(lambda: layout(prediction()), route="/prediction", title="Iris - Prediction")
app.add_page(lambda: layout(stats()), route="/stats", title="Iris - Stats")
app.add_page(lambda: layout(pca_page()), route="/pca", title="Iris - PCA")
app.add_page(lambda: layout(cluster()), route="/cluster", title="Iris - Cluster")
app.add_page(lambda: layout(network()), route="/network", title="Iris - Network")
app.add_page(lambda: layout(tree_page()), route="/tree", title="Iris - Tree")
app.add_page(lambda: layout(history()), route="/history", title="Iris - History")
app.add_page(lambda: layout(chat()), route="/chat", title="Iris - Chat")
app.add_page(lambda: layout(monitoring()), route="/monitoring", title="Iris - Monitoring")
