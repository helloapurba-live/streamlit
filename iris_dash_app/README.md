# 🌸 Iris Dash App

A production-ready Dash Plotly web application for Iris dataset analysis.

## 🚀 Quick Start

### Prerequisites
Make sure you're in the `ai310` conda environment:
```bash
conda activate ai310
```

### Install Dependencies
```bash
pip install dash dash-bootstrap-components pandas numpy scikit-learn matplotlib seaborn plotly sqlalchemy
```

### Optional Dependencies (for advanced features)
```bash
# For network visualizations (Pages 6-7)
conda install -y networkx scipy

# For AI Chat feature (Page 9)
pip install pandasai openai

# For MLOps monitoring (Page 11)
pip install evidently
```

### Run the App
```bash
python app.py
```

Then open your browser to: **http://localhost:8050**

## 📊 Features

### Core Pages (Always Available)
- 🏠 **Home** - Welcome page
- 📊 **General EDA** - Interactive histograms and box plots
- 🤖 **Prediction** - Real-time ML predictions with sliders
- 📉 **Statistical Insights** - Violin plots, parallel coordinates, correlation heatmap
- 🧠 **PCA Manifold** - 3D PCA visualization
- 🕸️ **Cluster Analysis** - K-Means clustering with interactive controls
- 🌲 **Decision Tree** - Tree visualization

### Advanced Pages (Require Optional Dependencies)
- 🔗 **KNN Network** - 3D network graph (requires `networkx`)
- 🕸️ **Feature Network** - Correlation network (requires `networkx`)
- 💬 **Chat with Data** - AI-powered chat (requires `pandasai` + OpenAI API key)
- 📜 **History** - Prediction history from database
- 🔍 **Model Monitoring** - Data drift detection (requires `evidently`)

## 🐛 Troubleshooting

### If you see "ModuleNotFoundError"
Install the missing package:
```bash
pip install <package-name>
```

### If network pages show errors
Install NetworkX:
```bash
conda install -y networkx
```

### If the app won't start
1. Make sure you're in the correct directory: `cd iris_dash_app`
2. Check Python version: `python --version` (should be 3.10+)
3. Verify Dash is installed: `python -c "import dash; print(dash.__version__)"`

## 🎯 Next Steps

- Try making predictions on Page 2
- Explore 3D visualizations on Pages 4-5
- View the correlation heatmap on Page 3
- Check prediction history on Page 10

Enjoy exploring the Iris dataset! 🌺
