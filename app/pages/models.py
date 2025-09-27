"""
Models page for the Commodity Price Predictor app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.data_loader import DataLoader
from services.features import FeatureEngineer
from services.modeling import ModelTrainer
from services.plots import PlotBuilder
from utils.constants import COMMODITIES, HORIZONS, MODEL_TYPES

def show_models():
    """Display the models page."""
    
    st.title("🤖 Model Training & Prediction")
    st.markdown("Train and evaluate prediction models for commodity prices.")
    
    # Get services from session state
    data_loader = st.session_state.data_loader
    feature_engineer = st.session_state.feature_engineer
    model_trainer = st.session_state.model_trainer
    plot_builder = st.session_state.plot_builder
    
    # Sidebar controls
    st.sidebar.subheader("Model Configuration")
    
    # Symbol selection
    symbols = list(COMMODITIES.keys())
    selected_symbols = st.sidebar.multiselect(
        "Select Commodities",
        symbols,
        default=symbols[:2]
    )
    
    # Horizon selection
    selected_horizons = st.sidebar.multiselect(
        "Select Horizons",
        list(HORIZONS.keys()),
        default=['1d', '5d']
    )
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select Models",
        list(MODEL_TYPES.keys()),
        default=['naive_last', 'arima', 'xgboost']
    )
    
    # Enable Prophet
    use_prophet = st.sidebar.checkbox("Enable Prophet (Optional)", value=False)
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    
    initial_train_days = st.sidebar.slider(
        "Initial Training Days",
        min_value=252,
        max_value=1000,
        value=500,
        step=50
    )
    
    step_days = st.sidebar.slider(
        "Step Days",
        min_value=1,
        max_value=20,
        value=5
    )
    
    # Load data
    if st.sidebar.button("🔄 Load Data"):
        with st.spinner("Loading data..."):
            data = data_loader.load_commodity_data(selected_symbols)
            st.session_state.data = data
            st.success(f"Loaded data for {len(data)} commodities")
    
    # Check if data is loaded
    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data first using the sidebar controls.")
        return
    
    data = st.session_state.data
    
    # Train Models
    st.subheader("🚀 Train Models")
    
    if st.button("🤖 Train All Models", type="primary"):
        if not selected_symbols or not selected_horizons or not selected_models:
            st.error("Please select symbols, horizons, and models to train.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = {}
        total_tasks = len(selected_symbols) * len(selected_horizons)
        current_task = 0
        
        for symbol in selected_symbols:
            if symbol not in data or data[symbol].empty:
                continue
            
            # Create features
            df = feature_engineer.create_features(data[symbol], symbol)
            df = feature_engineer.create_targets(df, selected_horizons)
            
            # Train models
            symbol_results = {}
            for horizon in selected_horizons:
                current_task += 1
                progress_bar.progress(current_task / total_tasks)
                status_text.text(f"Training {symbol} {horizon}...")
                
                try:
                    horizon_results = model_trainer.train_models(
                        df, symbol, [horizon], use_prophet
                    )
                    symbol_results[horizon] = horizon_results.get(horizon, {})
                except Exception as e:
                    st.error(f"Error training {symbol} {horizon}: {str(e)}")
                    symbol_results[horizon] = {'error': str(e)}
            
            all_results[symbol] = symbol_results
        
        st.session_state.model_results = all_results
        progress_bar.progress(1.0)
        status_text.text("Training completed!")
        st.success("All models trained successfully!")
    
    # Display Results
    if 'model_results' in st.session_state:
        st.subheader("📊 Model Results")
        
        model_results = st.session_state.model_results
        
        # Model comparison
        st.subheader("🏆 Model Comparison")
        
        comparison_data = []
        for symbol, symbol_results in model_results.items():
            for horizon, horizon_results in symbol_results.items():
                if 'error' in horizon_results:
                    continue
                
                metrics = horizon_results.get('metrics', {})
                for model_name, model_metrics in metrics.items():
                    if 'error' in model_metrics:
                        continue
                    
                    comparison_data.append({
                        'Symbol': symbol,
                        'Horizon': horizon,
                        'Model': model_name,
                        'RMSE': model_metrics.get('rmse', 0),
                        'MAE': model_metrics.get('mae', 0),
                        'R²': model_metrics.get('r2', 0),
                        'MAPE': model_metrics.get('mape', 0),
                        'Direction Accuracy': model_metrics.get('direction_accuracy', 0)
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Sort by R² score
            comparison_df = comparison_df.sort_values('R²', ascending=False)
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Best model per symbol/horizon
            st.subheader("🥇 Best Models")
            
            best_models = comparison_df.groupby(['Symbol', 'Horizon']).first().reset_index()
            st.dataframe(best_models[['Symbol', 'Horizon', 'Model', 'R²', 'Direction Accuracy']], use_container_width=True)
        
        # Individual Model Analysis
        st.subheader("🔍 Individual Model Analysis")
        
        # Select symbol and horizon for detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            selected_symbol = st.selectbox("Select Symbol", list(model_results.keys()))
        
        with col2:
            if selected_symbol in model_results:
                available_horizons = [h for h in model_results[selected_symbol].keys() 
                                    if 'error' not in model_results[selected_symbol][h]]
                selected_horizon = st.selectbox("Select Horizon", available_horizons)
            else:
                selected_horizon = None
        
        if selected_symbol and selected_horizon:
            horizon_results = model_results[selected_symbol][selected_horizon]
            
            if 'error' not in horizon_results:
                # Model metrics
                st.subheader(f"📈 {selected_symbol} - {selected_horizon} Results")
                
                metrics = horizon_results.get('metrics', {})
                
                # Display metrics in columns
                cols = st.columns(len(metrics))
                for i, (model_name, model_metrics) in enumerate(metrics.items()):
                    if 'error' in model_metrics:
                        continue
                    
                    with cols[i % len(cols)]:
                        st.metric(
                            f"{model_name}",
                            f"{model_metrics.get('r2', 0):.3f}",
                            f"RMSE: {model_metrics.get('rmse', 0):.4f}"
                        )
                
                # Model performance comparison
                st.subheader("📊 Performance Comparison")
                
                model_names = list(metrics.keys())
                model_names = [name for name in model_names if 'error' not in metrics.get(name, {})]
                
                if model_names:
                    # Create comparison chart
                    import plotly.express as px
                    
                    chart_data = []
                    for model_name in model_names:
                        model_metrics = metrics[model_name]
                        chart_data.append({
                            'Model': model_name,
                            'R²': model_metrics.get('r2', 0),
                            'RMSE': model_metrics.get('rmse', 0),
                            'MAE': model_metrics.get('mae', 0),
                            'Direction Accuracy': model_metrics.get('direction_accuracy', 0)
                        })
                    
                    chart_df = pd.DataFrame(chart_data)
                    
                    # R² comparison
                    fig_r2 = px.bar(
                        chart_df, 
                        x='Model', 
                        y='R²',
                        title=f"R² Score Comparison - {selected_symbol} {selected_horizon}"
                    )
                    st.plotly_chart(fig_r2, use_container_width=True)
                    
                    # Direction accuracy comparison
                    fig_acc = px.bar(
                        chart_df, 
                        x='Model', 
                        y='Direction Accuracy',
                        title=f"Direction Accuracy Comparison - {selected_symbol} {selected_horizon}"
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                # Feature importance (for XGBoost)
                if 'xgboost' in horizon_results.get('models', {}):
                    st.subheader("🔍 Feature Importance (XGBoost)")
                    
                    xgb_model = horizon_results['models']['xgboost']
                    if 'model' in xgb_model:
                        importance_df = model_trainer.get_feature_importance(xgb_model)
                        
                        if not importance_df.empty:
                            # Display top 20 features
                            top_features = importance_df.head(20)
                            
                            fig_importance = plot_builder.create_feature_importance_chart(
                                top_features,
                                f"Feature Importance - {selected_symbol} {selected_horizon}"
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Feature importance table
                            st.dataframe(importance_df, use_container_width=True)
    
    # Walk-Forward Validation
    st.subheader("🔄 Walk-Forward Validation")
    
    if st.button("🔄 Run Walk-Forward Validation"):
        if not selected_symbols or not selected_horizons:
            st.error("Please select symbols and horizons first.")
            return
        
        wf_results = {}
        
        for symbol in selected_symbols:
            if symbol not in data or data[symbol].empty:
                continue
            
            # Create features
            df = feature_engineer.create_features(data[symbol], symbol)
            df = feature_engineer.create_targets(df, selected_horizons)
            
            symbol_wf_results = {}
            for horizon in selected_horizons:
                try:
                    wf_result = model_trainer.walk_forward_validation(
                        df, symbol, horizon, initial_train_days, step_days
                    )
                    symbol_wf_results[horizon] = wf_result
                except Exception as e:
                    st.error(f"Error in walk-forward validation for {symbol} {horizon}: {str(e)}")
                    symbol_wf_results[horizon] = {'error': str(e)}
            
            wf_results[symbol] = symbol_wf_results
        
        st.session_state.wf_results = wf_results
        st.success("Walk-forward validation completed!")
    
    # Display Walk-Forward Results
    if 'wf_results' in st.session_state:
        st.subheader("📊 Walk-Forward Validation Results")
        
        wf_results = st.session_state.wf_results
        
        for symbol, symbol_results in wf_results.items():
            st.subheader(f"📈 {symbol}")
            
            for horizon, horizon_results in symbol_results.items():
                if 'error' in horizon_results:
                    st.error(f"Error in {horizon}: {horizon_results['error']}")
                    continue
                
                st.write(f"**{horizon}** - {horizon_results.get('n_steps', 0)} steps")
                
                avg_metrics = horizon_results.get('avg_metrics', {})
                if avg_metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg RMSE", f"{avg_metrics.get('rmse', 0):.4f}")
                    with col2:
                        st.metric("Avg MAE", f"{avg_metrics.get('mae', 0):.4f}")
                    with col3:
                        st.metric("Avg R²", f"{avg_metrics.get('r2', 0):.3f}")
                    with col4:
                        st.metric("Avg Direction Accuracy", f"{avg_metrics.get('direction_accuracy', 0):.1f}%")
    
    # Model Export
    st.subheader("💾 Export Results")
    
    if 'model_results' in st.session_state:
        if st.button("📥 Download Model Results"):
            # Create summary of all results
            export_data = []
            for symbol, symbol_results in st.session_state.model_results.items():
                for horizon, horizon_results in symbol_results.items():
                    if 'error' in horizon_results:
                        continue
                    
                    metrics = horizon_results.get('metrics', {})
                    for model_name, model_metrics in metrics.items():
                        if 'error' in model_metrics:
                            continue
                        
                        export_data.append({
                            'Symbol': symbol,
                            'Horizon': horizon,
                            'Model': model_name,
                            'RMSE': model_metrics.get('rmse', 0),
                            'MAE': model_metrics.get('mae', 0),
                            'R2': model_metrics.get('r2', 0),
                            'MAPE': model_metrics.get('mape', 0),
                            'Direction_Accuracy': model_metrics.get('direction_accuracy', 0)
                        })
            
            if export_data:
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name=f"model_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
