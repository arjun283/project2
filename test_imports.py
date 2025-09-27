#!/usr/bin/env python3
"""
Test script to check imports and dependencies
"""

print("Testing imports...")

try:
    import streamlit as st
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Streamlit import error: {e}")

try:
    import pandas as pd
    print("✅ Pandas imported successfully")
except ImportError as e:
    print(f"❌ Pandas import error: {e}")

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import error: {e}")

try:
    import plotly
    print("✅ Plotly imported successfully")
except ImportError as e:
    print(f"❌ Plotly import error: {e}")

try:
    from utils.constants import COMMODITIES
    print("✅ Utils constants imported successfully")
except ImportError as e:
    print(f"❌ Utils constants import error: {e}")

try:
    from services.data_loader import DataLoader
    print("✅ DataLoader imported successfully")
except ImportError as e:
    print(f"❌ DataLoader import error: {e}")

print("\nTesting basic Streamlit app...")

try:
    import streamlit as st
    
    st.title("Test App")
    st.write("If you can see this, Streamlit is working!")
    
    print("✅ Basic Streamlit app created successfully")
except Exception as e:
    print(f"❌ Streamlit app error: {e}")

print("\nAll tests completed!")
