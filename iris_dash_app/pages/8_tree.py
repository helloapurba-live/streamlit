import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from utils import get_model_and_data
import io
import base64

dash.register_page(__name__, name='🌲 Decision Tree', order=8)

clf, target_names, feature_names, X = get_model_and_data()

layout = dbc.Container([
    html.H2("🌲 Decision Tree Visualization"),
    html.Hr(),
    
    html.Div(
        html.Img(id='tree-img', style={'width': '100%'}),
        className="text-center"
    )
])

# Since the tree is static for the loaded model, we can compute it once or on load.
# We'll use a callback just to follow pattern, or compute immediately.
def get_tree_image():
    # Train a single tree for visualization (RandomForest is a collection)
    from sklearn.tree import DecisionTreeClassifier
    tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_clf.fit(X, clf.predict(X)) # mimic the RF behavior roughly or just fit on GT
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(tree_clf, feature_names=feature_names, class_names=target_names, filled=True, ax=ax)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    return "data:image/png;base64,{}".format(data)

layout.children[2].children.src = get_tree_image()
