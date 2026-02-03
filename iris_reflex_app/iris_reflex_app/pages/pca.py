import reflex as rx
import plotly.express as px
from sklearn.decomposition import PCA
from ..utils import load_data

def pca_page() -> rx.Component:
    df = load_data()
    features = df.columns[:-1]
    
    # PCA
    pca = PCA(n_components=3)
    components = pca.fit_transform(df[features])
    
    pca_df = px.pd.DataFrame(
        components, 
        columns=['PC1', 'PC2', 'PC3']
    )
    pca_df['species'] = df['species']
    
    fig = px.scatter_3d(
        pca_df, x='PC1', y='PC2', z='PC3',
        color='species', 
        symbol='species',
        title="3D PCA Manifold Projection"
    )

    return rx.vstack(
        rx.heading("🧠 PCA Manifold Visualization", size="8"),
        rx.divider(),
        rx.text("Reducing the 4D feature space into 3 components for intuitive visualization."),
        rx.card(
            rx.plotly(data=fig, height="600px"),
            width="100%",
        ),
        rx.callout(
            f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.2%}",
            icon="info",
            width="100%",
        ),
        spacing="6",
    )
