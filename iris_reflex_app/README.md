# 🌸 Iris Reflex App

A modern, full-stack Python web application for Iris dataset analysis using the [Reflex](https://reflex.dev) framework.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you are in the `ai310` conda environment:
```bash
conda activate ai310
```

### 2. Install Dependencies
```bash
pip install reflex pandas numpy scikit-learn plotly sqlalchemy matplotlib
```

### 3. Initialize & Run
```bash
# Navigate to the project folder
cd iris_reflex_app

# Run the app in development mode
reflex run
```

The app will be available at:
- **Frontend**: http://localhost:3000
- **Backend/API**: http://localhost:8000

## 📊 Features

- ✅ **Reactive UI**: Built with Chakra UI components
- ✅ **11 Interactive Pages**: EDA, Predictions, PCA, Clustering, etc.
- ✅ **Real-time ML**: Random Forest predictions with live slider updates
- ✅ **Advanced Visualizations**: 3D PCA, Violin plots, Parallel Coordinates
- ✅ **Shared Backend**: Connects to the same SQLite database as Streamlit/Dash

## 🏗️ Structure

- `iris_reflex_app.py`: Main app entry and page registration
- `state.py`: Reactive state and ML prediction logic
- `pages/`: Individual page components
- `utils.py`: Shared ML utilities

## 🛠️ Troubleshooting

- **Node.js/Bun Error**: Reflex requires Node.js or Bun. It usually installs them automatically, but ensure your environment has network access.
- **Port Conflict**: If port 3000 or 8000 is used, specify different ports: `reflex run --frontend-port 3001 --backend-port 8001`.

---
*Part of the Triple-Framework Iris ML Suite*
