@echo off
echo ===================================================
echo 🧹 Creating a clean Python environment for Iris Dash
echo ===================================================

:: 1. Create Virtual Environment
python -m venv dash_env

:: 2. Activate it
call dash_env\Scripts\activate

:: 3. Upgrade pip
python -m pip install --upgrade pip

:: 4. Install Dependencies (Cleanly)
echo ⬇️ Installing dependencies...
pip install dash dash-bootstrap-components pandas numpy scikit-learn matplotlib seaborn plotly networkx scipy sqlalchemy

:: Optional: Install complex production libs (might fail if compilation needed, so we do them last or skip if strict)
echo ⬇️ Installing AI extras...
pip install pandasai openai evidently

:: 5. Run the App
echo 🚀 Launching Dash App...
python iris_dash_app/app.py

echo 🚀 Dash App should be running now...
