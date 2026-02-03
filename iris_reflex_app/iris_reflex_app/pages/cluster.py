import reflex as rx
import plotly.express as px
from sklearn.cluster import KMeans
from ..utils import load_data

def cluster() -> rx.Component:
    df = load_data()
    features = [df.columns[0], df.columns[1]] # Sepal L vs W
    
    # Simple K-Means (fixed K for demo simplicity, can add state later if needed)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[features].values)
    
    fig = px.scatter(
        df, x=features[0], y=features[1],
        color='cluster', 
        hover_data=['species'],
        title="K-Means Clustering Analysis (K=3)"
    )

    return rx.vstack(
        rx.heading("🕸️ Cluster Analysis", size="8"),
        rx.divider(),
        rx.text("Unsupervised learning applied to find natural groupings in the data."),
        rx.card(
            rx.plotly(data=fig, height="500px"),
            width="100%",
        ),
        spacing="6",
    )
