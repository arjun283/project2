"""
Overview page for the Commodity Price Predictor app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.services.data_loader import DataLoader
from app.services.features import FeatureEngineer
from app.services.regimes import RegimeDetector
from app.services.modeling import ModelTrainer
from app.services.plots import PlotBuilder
from app.utils.constants import COMMODITIES, HORIZONS

def show_overview():
    """Display the overview page."""
    
    st.title("🏠 Overview")
    st.markdown("Welcome to the Commodity Price Predictor - your comprehensive tool for commodity market analysis and prediction.")
    
    # Get services from session state
    data_loader = st.session_state.data_loader
    feature_engineer = st.session_state.feature_engineer
    regime_detector = st.session_state.regime_detector
    model_trainer = st.session_state.model_trainer
    plot_builder = st.session_state.plot_builder
    
    # Load data
    with st.spinner("Loading commodity data..."):
        symbols = list(COMMODITIES.keys())
        data = data_loader.load_commodity_data(symbols)
        
        if not data:
            st.error("No data available. Please check your internet connection or data files.")
            return
    
    # Key Metrics Row
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Commodities", len(data))
    
    with col2:
        total_data_points = sum(len(df) for df in data.values())
        st.metric("Total Data Points", f"{total_data_points:,}")
    
    with col3:
        avg_vol = np.mean([df['Close'].pct_change().std() * np.sqrt(252) 
                          for df in data.values() if 'Close' in df.columns])
        st.metric("Avg Volatility", f"{avg_vol:.1%}")
    
    with col4:
        st.metric("Last Update", datetime.now().strftime("%Y-%m-%d"))
    
    # Market Overview
    st.subheader("🌍 Market Overview")
    
    # Create summary table
    summary_data = []
    for symbol, df in data.items():
        if df.empty or 'Close' not in df.columns:
            continue
        
        # Calculate metrics
        last_close = df['Close'].iloc[-1]
        returns_20d = df['Close'].pct_change(20).iloc[-1]
        vol_20d = df['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Detect regime
        df_with_regime = regime_detector.detect_regimes(df, symbol)
        current_regime = regime_detector.get_current_regime(symbol)
        
        summary_data.append({
            'Symbol': symbol,
            'Name': COMMODITIES[symbol]['name'],
            'Category': COMMODITIES[symbol]['category'],
            'Last Close': f"${last_close:.2f}",
            '20d Return': f"{returns_20d:.1%}",
            '20d Vol': f"{vol_20d:.1%}",
            'Regime': current_regime or 'Unknown'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True)
    
    # Price Charts
    st.subheader("📈 Price Charts")
    
    # Select commodities to display
    selected_symbols = st.multiselect(
        "Select commodities to display",
        list(data.keys()),
        default=list(data.keys())[:3]
    )
    
    if selected_symbols:
        # Create tabs for each commodity
        tabs = st.tabs(selected_symbols)
        
        for i, symbol in enumerate(selected_symbols):
            with tabs[i]:
                if symbol in data and not data[symbol].empty:
                    # Create price chart
                    fig = plot_builder.create_price_chart(
                        data[symbol], 
                        symbol,
                        show_volume=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show regime chart
                    df_with_regime = regime_detector.detect_regimes(data[symbol], symbol)
                    regime_fig = plot_builder.create_regime_chart(df_with_regime, symbol)
                    st.plotly_chart(regime_fig, use_container_width=True)
    
    # Model Performance Summary
    st.subheader("🤖 Model Performance Summary")
    
    # This would show recent model performance
    # For now, show placeholder
    st.info("Model performance metrics will be displayed here after training models.")
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            with st.spinner("Refreshing data..."):
                data = data_loader.refresh_data(symbols)
                st.success("Data refreshed successfully!")
                st.rerun()
    
    with col2:
        if st.button("🤖 Train Models", use_container_width=True):
            st.info("Navigate to the Models page to train prediction models.")
    
    with col3:
        if st.button("📊 Run Backtest", use_container_width=True):
            st.info("Navigate to the Backtest page to run strategy backtests.")
    
    # Data Quality Status
    st.subheader("🔍 Data Quality Status")
    
    quality_data = []
    for symbol, df in data.items():
        if df.empty:
            continue
        
        # Check data quality
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        data_points = len(df)
        last_date = pd.to_datetime(df['Date']).max()
        days_old = (datetime.now() - last_date).days
        
        quality_data.append({
            'Symbol': symbol,
            'Data Points': data_points,
            'Missing %': f"{missing_pct:.1f}%",
            'Last Date': last_date.strftime('%Y-%m-%d'),
            'Days Old': days_old,
            'Status': 'Good' if missing_pct < 5 and days_old < 7 else 'Issues'
        })
    
    quality_df = pd.DataFrame(quality_data)
    
    if not quality_df.empty:
        st.dataframe(quality_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Commodity Price Predictor** - Built with Streamlit, powered by machine learning")
