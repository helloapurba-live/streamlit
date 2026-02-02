import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, _tree

st.set_page_config(page_title="Decision Tree Graph", page_icon="🌲", layout="wide")

st.title("🌲 Decision Tree Graph")
st.markdown("Visualizing the **Logic Flow** of the AI. This fits a single decision tree and plots the hierarchy exactly as NetworkX sees it.")

iris = load_iris()
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(iris.data, iris.target)

def tree_to_networkx(clf, feature_names):
    G = nx.DiGraph()
    tree = clf.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.feature
    ]
    
    def recurse(node, parent=None):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree.threshold[node]
            label = f"{name} <= {threshold:.2f}"
            G.add_node(node, label=label, type='decision')
        else:
            # Leaf
            values = tree.value[node][0]
            class_idx = values.argmax()
            class_name = iris.target_names[class_idx]
            label = f"Class: {class_name}"
            G.add_node(node, label=label, type='leaf')
            
        if parent is not None:
            G.add_edge(parent, node)
            
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            recurse(tree.children_left[node], node)
            recurse(tree.children_right[node], node)

    recurse(0)
    return G

G = tree_to_networkx(clf, iris.feature_names)

# Hierarchy Layout (top-down)
# NetworkX 'dot' or 'multipartite' layout is hard without Graphviz.
# We will construct a manual hierarchy layout algorithm for simplicity in pure python.
pos = {}
def hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    # Minimal custom layout for trees
    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

pos = hierarchy_pos(G, 0)

# Draw
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    mode='lines',
    line=dict(width=1, color='#888'),
    hoverinfo='none'
))

node_x = []
node_y = []
node_text = []
node_colors = []

for node_id, attrs in G.nodes(data=True):
    x, y = pos[node_id]
    node_x.append(x)
    node_y.append(y)
    node_text.append(attrs['label'])
    if attrs['type'] == 'decision':
        node_colors.append('#FFA07A') # Light Salmon
    else:
        node_colors.append('#98FB98') # Pale Green

fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition="top center",
    marker=dict(
        size=15,
        color=node_colors,
        line=dict(width=2, color='#333')
    )
))

fig.update_layout(title="Decision Tree Flowchart", showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
st.plotly_chart(fig, use_container_width=True)
