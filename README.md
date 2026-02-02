# 🌸 Interactive Iris Data Science Suite

A robust, multipage **Streamlit** application designed to explore, analyze, and model the famous Iris flower dataset. This project demonstrates advanced data visualization techniques, machine learning workflows, and network analysis.

## 🚀 Features

The application is structured into **8 different modules**, covering the entire Data Science lifecycle:

### 🏠 Home
-   **Landing Page**: Overview of the project, dataset, and navigation guide.

### 📊 Exploratory Data Analysis (EDA)
-   **Page 1: General EDA**: Interactive histograms, box plots, scatter plots, and simple 3D projections.
-   **Page 3: Statistical Insights**: Violin plots for distribution density and Parallel Coordinates for multivariate separation.
-   **Page 4: PCA & Manifold**: Live Dimensionality Reduction (PCA) visualizing 4D data in 2D and 3D space.
-   **Page 5: Cluster Analysis**: Unsupervised Learning (K-Means) playground. Try to "rediscover" the species without labels!

### 🕸️ Graph & Network Analysis
-   **Page 6: KNN Network**: A 3D topological view of the data where samples are nodes connected to their nearest neighbors.
-   **Page 7: Feature Network**: "the Internet of Variables" - visualizing how strongly different features correlate with each other.
-   **Page 8: Decision Tree Graph**: A functional flowchart showing exactly how the Model makes its decisions (White-box AI).

### 🤖 Modeling
-   **Page 2: Model Prediction**: A production-ready inference engine. Adjust sliders to inputs and get real-time predictions with confidence scores.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/iris-streamlit-app.git
    cd iris-streamlit-app
    ```

2.  **Create a virtual environment (Optional but Recommended)**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Navigate through the pages using the Sidebar on the left.

## 📦 Dependencies

*   **Streamlit**: Web App Framework
*   **Pandas/Numpy**: Data Manipulation
*   **Scikit-Learn**: Machine Learning Models (Random Forest, K-Means, PCA, KNN)
*   **Plotly**: Interactive Visualizations
*   **Matplotlib/Seaborn**: Static Statistical Plots
*   **NetworkX**: Graph & Network Algorithms
*   **Scipy**: Scientific Computing (required for adjacency matrices)

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
