#!/usr/bin/env python
"""Test script to diagnose Dash app issues"""

print("Testing imports...")

try:
    import dash
    print("✓ dash imported successfully")
except Exception as e:
    print(f"✗ dash import failed: {e}")

try:
    import dash_bootstrap_components as dbc
    print("✓ dash_bootstrap_components imported successfully")
except Exception as e:
    print(f"✗ dash_bootstrap_components import failed: {e}")

try:
    from utils import load_data
    print("✓ utils.load_data imported successfully")
    df = load_data()
    print(f"✓ Data loaded: {len(df)} rows")
except Exception as e:
    print(f"✗ utils import failed: {e}")

print("\nTrying to create Dash app...")
try:
    from dash import Dash
    app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])
    print("✓ Basic Dash app created successfully")
except Exception as e:
    print(f"✗ Dash app creation failed: {e}")

print("\nTrying with use_pages=True...")
try:
    app2 = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB])
    print("✓ Dash app with pages created successfully")
except Exception as e:
    print(f"✗ Dash app with pages failed: {e}")
    import traceback
    traceback.print_exc()
