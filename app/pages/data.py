"""
Data page for the Commodity Price Predictor app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.data_loader import DataLoader
from services.features import FeatureEngineer
from services.regimes import RegimeDetector
from services.plots import PlotBuilder
from utils.constants import COMMODITIES, MACRO_INDICATORS

def show_data():
    """Display the data page."""
    
    st.title("📊 Data Analysis")
    st.markdown("Explore commodity price data, features, and market regimes.")
    
    # Get services from session state
    data_loader = st.session_state.data_loader
    feature_engineer = st.session_state.feature_engineer
    regime_detector = st.session_state.regime_detector
    plot_builder = st.session_state.plot_builder
    
    # Sidebar controls
    st.sidebar.subheader("Data Controls")
    
    # Symbol selection
    symbols = list(COMMODITIES.keys())
    selected_symbols = st.sidebar.multiselect(
        "Select Commodities",
        symbols,
        default=symbols[:3]
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Load data
    if st.sidebar.button("🔄 Load Data"):
        with st.spinner("Loading data..."):
            data = data_loader.load_commodity_data(
                selected_symbols,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            st.session_state.data = data
            st.success(f"Loaded data for {len(data)} commodities")
    
    # Load macro data
    if st.sidebar.button("📈 Load Macro Data"):
        with st.spinner("Loading macro data..."):
            macro_data = data_loader.load_macro_data(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            st.session_state.macro_data = macro_data
            st.success(f"Loaded macro data for {len(macro_data)} indicators")
    
    # Check if data is loaded
    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data first using the sidebar controls.")
        return
    
    data = st.session_state.data
    macro_data = st.session_state.get('macro_data', {})
    
    # Data Summary
    st.subheader("📋 Data Summary")
    
    summary_data = []
    for symbol, df in data.items():
        if df.empty:
            continue
        
        summary_data.append({
            'Symbol': symbol,
            'Name': COMMODITIES[symbol]['name'],
            'Category': COMMODITIES[symbol]['category'],
            'Start Date': df['Date'].min(),
            'End Date': df['Date'].max(),
            'Data Points': len(df),
            'Missing %': f"{(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%",
            'Last Close': f"${df['Close'].iloc[-1]:.2f}" if 'Close' in df.columns else "N/A"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Individual Commodity Analysis
    st.subheader("🔍 Individual Analysis")
    
    selected_symbol = st.selectbox("Select Commodity", list(data.keys()))
    
    if selected_symbol in data:
        df = data[selected_symbol]
        
        if not df.empty:
            # Create features
            df_with_features = feature_engineer.create_features(df, selected_symbol, macro_data)
            
            # Detect regimes
            df_with_regime = regime_detector.detect_regimes(df_with_features, selected_symbol)
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Returns Analysis", "Regime Analysis", "Features"])
            
            with tab1:
                # Price chart
                fig = plot_builder.create_price_chart(df_with_regime, selected_symbol)
                st.plotly_chart(fig, use_container_width=True)
                
                # Price statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${df_with_regime['Close'].iloc[-1]:.2f}")
                with col2:
                    returns_1d = df_with_regime['Close'].pct_change().iloc[-1]
                    st.metric("1d Return", f"{returns_1d:.2%}")
                with col3:
                    returns_20d = df_with_regime['Close'].pct_change(20).iloc[-1]
                    st.metric("20d Return", f"{returns_20d:.2%}")
                with col4:
                    vol_20d = df_with_regime['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
                    st.metric("20d Volatility", f"{vol_20d:.1%}")
            
            with tab2:
                # Returns analysis
                returns_fig = plot_builder.create_returns_chart(df_with_regime, selected_symbol)
                st.plotly_chart(returns_fig, use_container_width=True)
                
                # Returns statistics
                returns = df_with_regime['returns'].dropna()
                if not returns.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Return", f"{returns.mean():.4f}")
                    with col2:
                        st.metric("Std Return", f"{returns.std():.4f}")
                    with col3:
                        st.metric("Skewness", f"{returns.skew():.2f}")
                    with col4:
                        st.metric("Kurtosis", f"{returns.kurtosis():.2f}")
            
            with tab3:
                # Regime analysis
                regime_fig = plot_builder.create_regime_chart(df_with_regime, selected_symbol)
                st.plotly_chart(regime_fig, use_container_width=True)
                
                # Current regime
                current_regime = regime_detector.get_current_regime(selected_symbol)
                st.info(f"Current Regime: **{current_regime}**")
                
                # Regime statistics
                regime_stats = regime_detector.get_regime_stats(selected_symbol)
                if regime_stats:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Regime Counts")
                        regime_counts = regime_stats.get('regime_counts', {})
                        for regime, count in regime_counts.items():
                            st.write(f"**{regime}**: {count} days")
                    
                    with col2:
                        st.subheader("Regime Durations")
                        durations = regime_stats.get('regime_durations', [])
                        for duration in durations[-5:]:  # Last 5 regimes
                            st.write(f"**{duration['regime']}**: {duration['duration_days']} days")
            
            with tab4:
                # Features analysis
                st.subheader("Feature Statistics")
                
                # Get feature statistics
                feature_stats = feature_engineer.get_feature_statistics(df_with_features)
                
                if not feature_stats.empty:
                    st.dataframe(feature_stats, use_container_width=True)
                else:
                    st.info("No feature statistics available")
                
                # Feature correlation
                st.subheader("Feature Correlation")
                
                # Select features for correlation
                numeric_cols = df_with_features.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
                
                if len(feature_cols) > 1:
                    selected_features = st.multiselect(
                        "Select features for correlation",
                        feature_cols,
                        default=feature_cols[:10]
                    )
                    
                    if selected_features:
                        corr_data = df_with_features[selected_features].corr()
                        
                        # Create correlation heatmap
                        import plotly.express as px
                        fig = px.imshow(
                            corr_data,
                            text_auto=True,
                            aspect="auto",
                            title="Feature Correlation Matrix"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # Cross-Asset Analysis
    if len(data) > 1:
        st.subheader("🌐 Cross-Asset Analysis")
        
        # Returns correlation
        returns_data = {}
        for symbol, df in data.items():
            if 'returns' in df.columns:
                returns_data[symbol] = df['returns'].dropna()
        
        if returns_data:
            corr_fig = plot_builder.create_correlation_heatmap(returns_data)
            st.plotly_chart(corr_fig, use_container_width=True)
            
            # Correlation table
            returns_df = pd.DataFrame(returns_data)
            corr_matrix = returns_df.corr()
            
            st.subheader("Returns Correlation Matrix")
            st.dataframe(corr_matrix.round(3), use_container_width=True)
    
    # Data Export
    st.subheader("💾 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download All Data as CSV"):
            # Combine all data
            all_data = []
            for symbol, df in data.items():
                df_copy = df.copy()
                df_copy['Symbol'] = symbol
                all_data.append(df_copy)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"commodity_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("📊 Download Summary as CSV"):
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary",
                data=csv,
                file_name=f"commodity_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
