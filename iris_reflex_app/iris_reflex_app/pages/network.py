import reflex as rx
import plotly.graph_objects as go
import numpy as np
from ..utils import load_data

def network() -> rx.Component:
    df = load_data()
    corr = df.iloc[:, :-1].corr().values
    
    # Simple network visualization (Nodes in a circle)
    names = df.columns[:-1]
    n = len(names)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    
    edge_x = []
    edge_y = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr[i, j]) > 0.5:
                edge_x.extend([x[i], x[j], None])
                edge_y.extend([y[i], y[j], None])
                
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+text',
        text=names,
        textposition="top center",
        marker=dict(size=20, color='skyblue', line_width=2)
    ))
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title="Feature Correlation Network (threshold > 0.5)"
    )

    return rx.vstack(
        rx.heading("🕸️ Feature Network Analysis", size="8"),
        rx.divider(),
        rx.card(
            rx.plotly(data=fig, height="500px"),
            width="100%",
        ),
        spacing="6",
    )
