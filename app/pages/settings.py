"""
Settings page for the Commodity Price Predictor app.
"""

import streamlit as st
import os
from datetime import datetime
from services.api import initialize_api, get_api_status, start_api_server, stop_api_server
from utils.constants import COMMODITIES, HORIZONS, MODEL_TYPES

def show_settings():
    """Display the settings page."""
    
    st.title("⚙️ Settings")
    st.markdown("Configure application settings and preferences.")
    
    # General Settings
    st.subheader("🔧 General Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Settings")
        
        # Data refresh frequency
        refresh_frequency = st.selectbox(
            "Data Refresh Frequency",
            ["1 hour", "6 hours", "12 hours", "24 hours", "Manual"],
            index=3
        )
        
        # Cache settings
        use_cache = st.checkbox("Use Data Cache", value=True)
        cache_ttl = st.slider("Cache TTL (hours)", 1, 24, 6)
        
        # Data quality thresholds
        st.subheader("Data Quality Thresholds")
        
        max_missing_pct = st.slider(
            "Max Missing Data %",
            1, 20, 10
        )
        
        min_data_points = st.slider(
            "Min Data Points",
            50, 500, 100
        )
    
    with col2:
        st.subheader("Model Settings")
        
        # Default model parameters
        default_model = st.selectbox(
            "Default Model",
            list(MODEL_TYPES.keys()),
            index=2  # XGBoost
        )
        
        # Training parameters
        st.subheader("Training Parameters")
        
        initial_train_days = st.slider(
            "Initial Training Days",
            252, 1000, 500
        )
        
        step_days = st.slider(
            "Step Days",
            1, 20, 5
        )
        
        # Transaction costs
        st.subheader("Transaction Costs")
        
        default_transaction_cost = st.slider(
            "Default Transaction Cost (bps)",
            0, 50, 5
        )
    
    # API Settings
    st.subheader("🌐 API Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("REST API Configuration")
        
        # Check if API is available
        api_status = get_api_status()
        
        if api_status.get('api_available', False):
            st.success("✅ FastAPI is available")
            
            # API server status
            if api_status.get('server_running', False):
                st.success("🟢 API Server is running")
            else:
                st.warning("🔴 API Server is not running")
            
            # API controls
            if st.button("🚀 Start API Server"):
                try:
                    # Initialize API if not already done
                    if 'api_handler' not in st.session_state:
                        from services.data_loader import DataLoader
                        from services.modeling import ModelTrainer
                        
                        data_loader = DataLoader()
                        model_trainer = ModelTrainer()
                        api_handler = initialize_api(data_loader, model_trainer)
                        st.session_state.api_handler = api_handler
                    
                    result = start_api_server()
                    st.success(result.get('message', 'API server started'))
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error starting API server: {str(e)}")
            
            if st.button("🛑 Stop API Server"):
                try:
                    result = stop_api_server()
                    st.success(result.get('message', 'API server stopped'))
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error stopping API server: {str(e)}")
            
            # API documentation links
            st.subheader("📚 API Documentation")
            
            if api_status.get('server_running', False):
                st.markdown(f"**API Docs:** http://127.0.0.1:8000/docs")
                st.markdown(f"**ReDoc:** http://127.0.0.1:8000/redoc")
            else:
                st.info("Start the API server to access documentation")
        
        else:
            st.error("❌ FastAPI is not available. Install with: pip install fastapi uvicorn")
    
    with col2:
        st.subheader("API Endpoints")
        
        if api_status.get('server_running', False):
            st.markdown("""
            **Available Endpoints:**
            - `GET /` - Root endpoint
            - `GET /health` - Health check
            - `POST /predict` - Make predictions
            - `GET /symbols` - Get available symbols
            - `GET /horizons` - Get available horizons
            - `GET /models` - Get available models
            - `POST /train` - Train models
            """)
        else:
            st.info("Start the API server to see available endpoints")
    
    # Environment Variables
    st.subheader("🔐 Environment Variables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("API Keys")
        
        # FRED API Key
        fred_key = st.text_input(
            "FRED API Key",
            value=os.getenv('FRED_API_KEY', ''),
            type='password',
            help="Enter your FRED API key for macroeconomic data"
        )
        
        if fred_key and fred_key != os.getenv('FRED_API_KEY', ''):
            os.environ['FRED_API_KEY'] = fred_key
            st.success("FRED API key updated")
        
        # Random seed
        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=int(os.getenv('RANDOM_SEED', 42)),
            help="Random seed for reproducibility"
        )
        
        if random_seed != int(os.getenv('RANDOM_SEED', 42)):
            os.environ['RANDOM_SEED'] = str(random_seed)
            st.success("Random seed updated")
    
    with col2:
        st.subheader("Data Sources")
        
        # Data source preferences
        primary_source = st.selectbox(
            "Primary Data Source",
            ["yfinance", "cached"],
            help="Primary source for commodity data"
        )
        
        fallback_source = st.selectbox(
            "Fallback Data Source",
            ["cached", "yfinance"],
            help="Fallback source when primary fails"
        )
        
        # Macro data settings
        enable_macro = st.checkbox(
            "Enable Macro Data",
            value=True,
            help="Enable macroeconomic indicators"
        )
        
        macro_indicators = st.multiselect(
            "Macro Indicators",
            ["UUP", "SPY", "DGS10"],
            default=["UUP", "SPY"],
            help="Select macroeconomic indicators to use"
        )
    
    # Display Settings
    st.subheader("🎨 Display Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Chart Settings")
        
        # Chart theme
        chart_theme = st.selectbox(
            "Chart Theme",
            ["plotly_white", "plotly_dark", "ggplot2", "seaborn"],
            index=0
        )
        
        # Chart height
        chart_height = st.slider(
            "Default Chart Height",
            300, 800, 400
        )
        
        # Show volume
        show_volume = st.checkbox(
            "Show Volume Charts",
            value=True
        )
    
    with col2:
        st.subheader("Table Settings")
        
        # Table height
        table_height = st.slider(
            "Max Table Height",
            200, 600, 400
        )
        
        # Decimal places
        decimal_places = st.slider(
            "Decimal Places",
            2, 6, 4
        )
        
        # Show row numbers
        show_row_numbers = st.checkbox(
            "Show Row Numbers",
            value=False
        )
    
    # Performance Settings
    st.subheader("⚡ Performance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Caching")
        
        # Cache settings
        enable_caching = st.checkbox(
            "Enable Caching",
            value=True,
            help="Enable data and model caching"
        )
        
        cache_size = st.slider(
            "Cache Size (MB)",
            100, 1000, 500
        )
        
        # Cache TTL
        data_cache_ttl = st.slider(
            "Data Cache TTL (hours)",
            1, 24, 6
        )
        
        model_cache_ttl = st.slider(
            "Model Cache TTL (hours)",
            1, 168, 24
        )
    
    with col2:
        st.subheader("Processing")
        
        # Parallel processing
        enable_parallel = st.checkbox(
            "Enable Parallel Processing",
            value=True,
            help="Enable parallel processing for model training"
        )
        
        max_workers = st.slider(
            "Max Workers",
            1, 8, 4
        )
        
        # Memory limit
        memory_limit = st.slider(
            "Memory Limit (GB)",
            1, 16, 4
        )
    
    # Save Settings
    st.subheader("💾 Save Settings")
    
    if st.button("💾 Save All Settings", type="primary"):
        # Create settings dictionary
        settings = {
            'refresh_frequency': refresh_frequency,
            'use_cache': use_cache,
            'cache_ttl': cache_ttl,
            'max_missing_pct': max_missing_pct,
            'min_data_points': min_data_points,
            'default_model': default_model,
            'initial_train_days': initial_train_days,
            'step_days': step_days,
            'default_transaction_cost': default_transaction_cost,
            'fred_key': fred_key,
            'random_seed': random_seed,
            'primary_source': primary_source,
            'fallback_source': fallback_source,
            'enable_macro': enable_macro,
            'macro_indicators': macro_indicators,
            'chart_theme': chart_theme,
            'chart_height': chart_height,
            'show_volume': show_volume,
            'table_height': table_height,
            'decimal_places': decimal_places,
            'show_row_numbers': show_row_numbers,
            'enable_caching': enable_caching,
            'cache_size': cache_size,
            'data_cache_ttl': data_cache_ttl,
            'model_cache_ttl': model_cache_ttl,
            'enable_parallel': enable_parallel,
            'max_workers': max_workers,
            'memory_limit': memory_limit
        }
        
        # Save to session state
        st.session_state.settings = settings
        
        st.success("Settings saved successfully!")
    
    # Load Settings
    if 'settings' in st.session_state:
        st.subheader("📋 Current Settings")
        
        settings = st.session_state.settings
        
        # Display current settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Settings:**")
            st.write(f"- Refresh Frequency: {settings.get('refresh_frequency', 'N/A')}")
            st.write(f"- Use Cache: {settings.get('use_cache', 'N/A')}")
            st.write(f"- Cache TTL: {settings.get('cache_ttl', 'N/A')} hours")
            st.write(f"- Max Missing %: {settings.get('max_missing_pct', 'N/A')}%")
            st.write(f"- Min Data Points: {settings.get('min_data_points', 'N/A')}")
        
        with col2:
            st.markdown("**Model Settings:**")
            st.write(f"- Default Model: {settings.get('default_model', 'N/A')}")
            st.write(f"- Initial Train Days: {settings.get('initial_train_days', 'N/A')}")
            st.write(f"- Step Days: {settings.get('step_days', 'N/A')}")
            st.write(f"- Transaction Cost: {settings.get('default_transaction_cost', 'N/A')} bps")
            st.write(f"- Random Seed: {settings.get('random_seed', 'N/A')}")
    
    # Reset Settings
    st.subheader("🔄 Reset Settings")
    
    if st.button("🔄 Reset to Defaults"):
        # Clear session state settings
        if 'settings' in st.session_state:
            del st.session_state.settings
        
        st.success("Settings reset to defaults!")
        st.rerun()
    
    # Export/Import Settings
    st.subheader("📤 Export/Import Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'settings' in st.session_state:
            import json
            
            settings_json = json.dumps(st.session_state.settings, indent=2)
            
            st.download_button(
                label="📥 Download Settings",
                data=settings_json,
                file_name=f"settings_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_file = st.file_uploader(
            "📤 Upload Settings",
            type=['json'],
            help="Upload a settings JSON file"
        )
        
        if uploaded_file is not None:
            try:
                import json
                settings = json.load(uploaded_file)
                st.session_state.settings = settings
                st.success("Settings imported successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error importing settings: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Settings saved in session state. Restart the app to apply all changes.**")
