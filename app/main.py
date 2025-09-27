"""
Commodity Price Predictor - Main Streamlit App
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent
sys.path.append(str(app_dir))

# Import services
from app.services.data_loader import DataLoader
from app.services.features import FeatureEngineer
from app.services.regimes import RegimeDetector
from app.services.modeling import ModelTrainer
from app.services.backtest import BacktestEngine
from app.services.strategy import StrategySimulator
from app.services.explain import ModelExplainer
from app.services.plots import PlotBuilder
from app.services.api import initialize_api, get_api_status
from app.utils.constants import COMMODITIES, HORIZONS, MODEL_TYPES
from app.utils.seeds import set_seeds

# Page configuration
st.set_page_config(
    page_title="Commodity Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'feature_engineer' not in st.session_state:
    st.session_state.feature_engineer = FeatureEngineer()
if 'regime_detector' not in st.session_state:
    st.session_state.regime_detector = RegimeDetector()
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = ModelTrainer()
if 'backtest_engine' not in st.session_state:
    st.session_state.backtest_engine = BacktestEngine()
if 'strategy_simulator' not in st.session_state:
    st.session_state.strategy_simulator = StrategySimulator()
if 'model_explainer' not in st.session_state:
    st.session_state.model_explainer = ModelExplainer()
if 'plot_builder' not in st.session_state:
    st.session_state.plot_builder = PlotBuilder()

# Set random seeds
set_seeds()

def main():
    """Main application function."""
    
    # Sidebar
    st.sidebar.title("📈 Commodity Price Predictor")
    st.sidebar.markdown("---")
    
    # Navigation
    pages = {
        "🏠 Overview": "1_Overview",
        "📊 Data": "2_Data", 
        "🤖 Models": "3_Models",
        "📈 Backtest": "4_Backtest",
        "⚡ Strategy": "5_Strategy",
        "💼 Portfolio": "6_Portfolio",
        "🔍 Explainability": "7_Explainability",
        "🎯 Scenarios": "8_Scenarios",
        "⚙️ Settings": "9_Settings"
    }
    
    selected_page = st.sidebar.selectbox("Navigate", list(pages.keys()))
    
    # Load the selected page
    page_module = pages[selected_page]
    
    try:
        if page_module == "1_Overview":
            from pages.overview import show_overview
            show_overview()
        elif page_module == "2_Data":
            from pages.data import show_data
            show_data()
        elif page_module == "3_Models":
            from pages.models import show_models
            show_models()
        elif page_module == "4_Backtest":
            from pages.backtest import show_backtest
            show_backtest()
        elif page_module == "5_Strategy":
            from pages.strategy import show_strategy
            show_strategy()
        elif page_module == "6_Portfolio":
            from pages.portfolio import show_portfolio
            show_portfolio()
        elif page_module == "7_Explainability":
            from pages.explainability import show_explainability
            show_explainability()
        elif page_module == "8_Scenarios":
            from pages.scenarios import show_scenarios
            show_scenarios()
        elif page_module == "9_Settings":
            from pages.settings import show_settings
            show_settings()
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
