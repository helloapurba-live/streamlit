import reflex as rx

config = rx.Config(
    app_name="iris_reflex_app",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)