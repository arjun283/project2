"""
Explainability page for the Commodity Price Predictor app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.data_loader import DataLoader
from services.features import FeatureEngineer
from services.modeling import ModelTrainer
from services.explain import ModelExplainer
from services.plots import PlotBuilder
from utils.constants import COMMODITIES, HORIZONS

def show_explainability():
    """Display the explainability page."""
    
    st.title("🔍 Model Explainability")
    st.markdown("Understand how your models make predictions with SHAP analysis and feature importance.")
    
    # Get services from session state
    data_loader = st.session_state.data_loader
    feature_engineer = st.session_state.feature_engineer
    model_trainer = st.session_state.model_trainer
    model_explainer = st.session_state.model_explainer
    plot_builder = st.session_state.plot_builder
    
    # Sidebar controls
    st.sidebar.subheader("Model Selection")
    
    # Symbol selection
    symbols = list(COMMODITIES.keys())
    selected_symbol = st.sidebar.selectbox("Select Commodity", symbols)
    
    # Horizon selection
    selected_horizon = st.sidebar.selectbox("Select Horizon", list(HORIZONS.keys()))
    
    # Model selection
    model_name = st.sidebar.selectbox("Select Model", ["xgboost", "arima", "linear"])
    
    # Load data
    if st.sidebar.button("🔄 Load Data"):
        with st.spinner("Loading data..."):
            data = data_loader.load_commodity_data([selected_symbol])
            st.session_state.data = data
            st.success(f"Loaded data for {selected_symbol}")
    
    # Check if data is loaded
    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data first using the sidebar controls.")
        return
    
    data = st.session_state.data
    
    if selected_symbol not in data or data[selected_symbol].empty:
        st.error(f"No data available for {selected_symbol}")
        return
    
    df = data[selected_symbol]
    
    # Create features
    df_with_features = feature_engineer.create_features(df, selected_symbol)
    df_with_features = feature_engineer.create_targets(df_with_features, [selected_horizon])
    
    # Train model for explainability
    st.subheader("🤖 Train Model for Explainability")
    
    if st.button("🚀 Train Model", type="primary"):
        try:
            # Train model
            results = model_trainer.train_models(
                df_with_features, selected_symbol, [selected_horizon]
            )
            
            if selected_horizon in results and 'error' not in results[selected_horizon]:
                st.session_state.model_results = results
                st.success("Model trained successfully!")
            else:
                st.error("Model training failed.")
                
        except Exception as e:
            st.error(f"Model training error: {str(e)}")
    
    # Check if model is trained
    if 'model_results' not in st.session_state:
        st.warning("Please train a model first.")
        return
    
    model_results = st.session_state.model_results
    horizon_results = model_results.get(selected_horizon, {})
    
    if 'error' in horizon_results:
        st.error(f"Model error: {horizon_results['error']}")
        return
    
    # Get model and data for explainability
    models = horizon_results.get('models', {})
    if model_name not in models or 'error' in models[model_name]:
        st.error(f"Model {model_name} not available or has errors.")
        return
    
    model_data = models[model_name]
    if 'model' not in model_data:
        st.error(f"Model {model_name} not properly trained.")
        return
    
    model = model_data['model']
    
    # Prepare features for explainability
    feature_cols = [col for col in df_with_features.columns if col not in [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'data_quality',
        'target_1d', 'target_5d', 'target_20d', 'price_1d', 'price_5d', 'price_20d',
        'direction_1d', 'direction_5d', 'direction_20d', 'regime', 'regime_change'
    ]]
    
    X = df_with_features[feature_cols].fillna(0)
    y = df_with_features[f'target_{selected_horizon}'].dropna()
    
    # Align X and y
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    if X.empty or y.empty:
        st.error("No valid data for explainability analysis.")
        return
    
    # SHAP Analysis
    st.subheader("🔍 SHAP Analysis")
    
    if st.button("🔍 Run SHAP Analysis", type="primary"):
        try:
            # Run SHAP analysis
            explanations = model_explainer.explain_model(
                model, X, model_name, max_samples=100
            )
            
            if 'error' not in explanations:
                st.session_state.explanations = explanations
                st.success("SHAP analysis completed!")
            else:
                st.error(f"SHAP analysis error: {explanations['error']}")
                
        except Exception as e:
            st.error(f"SHAP analysis error: {str(e)}")
    
    # Display SHAP Results
    if 'explanations' in st.session_state:
        st.subheader("📊 SHAP Results")
        
        explanations = st.session_state.explanations
        
        if 'error' in explanations:
            st.error(f"Error: {explanations['error']}")
            return
        
        # Feature importance
        st.subheader("📈 Feature Importance")
        
        feature_importance = explanations.get('feature_importance', pd.DataFrame())
        
        if not feature_importance.empty:
            # Display top 20 features
            top_features = feature_importance.head(20)
            
            fig_importance = plot_builder.create_feature_importance_chart(
                top_features,
                f"Feature Importance - {selected_symbol} {selected_horizon}"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Feature importance table
            st.dataframe(feature_importance, use_container_width=True)
        
        # SHAP summary plot
        st.subheader("📊 SHAP Summary Plot")
        
        summary_data = explanations.get('summary_data', {})
        
        if summary_data:
            fig_summary = plot_builder.create_shap_summary_chart(
                summary_data,
                f"SHAP Summary - {selected_symbol} {selected_horizon}"
            )
            st.plotly_chart(fig_summary, use_container_width=True)
        
        # SHAP waterfall plot
        st.subheader("🌊 SHAP Waterfall Plot")
        
        waterfall_data = explanations.get('waterfall_data', {})
        
        if waterfall_data:
            fig_waterfall = plot_builder.create_waterfall_chart(
                waterfall_data,
                f"SHAP Waterfall - {selected_symbol} {selected_horizon}"
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Model Diagnostics
    st.subheader("🔍 Model Diagnostics")
    
    if st.button("🔍 Run Model Diagnostics", type="primary"):
        try:
            # Run diagnostics
            diagnostics = model_explainer.get_model_diagnostics(model, X, y)
            
            if 'error' not in diagnostics:
                st.session_state.diagnostics = diagnostics
                st.success("Model diagnostics completed!")
            else:
                st.error(f"Diagnostics error: {diagnostics['error']}")
                
        except Exception as e:
            st.error(f"Diagnostics error: {str(e)}")
    
    # Display Diagnostics
    if 'diagnostics' in st.session_state:
        st.subheader("📊 Model Diagnostics Results")
        
        diagnostics = st.session_state.diagnostics
        
        if 'error' in diagnostics:
            st.error(f"Error: {diagnostics['error']}")
            return
        
        # Basic diagnostics
        basic_diagnostics = diagnostics.get('diagnostics', {})
        
        if basic_diagnostics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R² Score", f"{basic_diagnostics.get('r_squared', 0):.3f}")
            with col2:
                st.metric("Mean Residual", f"{basic_diagnostics.get('mean_residual', 0):.4f}")
            with col3:
                st.metric("Residual Std", f"{basic_diagnostics.get('std_residual', 0):.4f}")
            with col4:
                st.metric("Residual Skewness", f"{basic_diagnostics.get('residual_skewness', 0):.3f}")
        
        # Residual analysis
        residual_analysis = diagnostics.get('residual_analysis', {})
        
        if residual_analysis:
            st.subheader("📊 Residual Analysis")
            
            residuals = residual_analysis.get('residuals', [])
            predictions = residual_analysis.get('predictions', [])
            actual = residual_analysis.get('actual', [])
            
            if residuals and predictions and actual:
                # Create residual plot
                import plotly.express as px
                
                residual_df = pd.DataFrame({
                    'Predictions': predictions,
                    'Residuals': residuals,
                    'Actual': actual
                })
                
                # Residuals vs Predictions
                fig_residuals = px.scatter(
                    residual_df,
                    x='Predictions',
                    y='Residuals',
                    title="Residuals vs Predictions"
                )
                st.plotly_chart(fig_residuals, use_container_width=True)
                
                # Q-Q plot
                from scipy import stats
                qq_data = stats.probplot(residuals, dist="norm")
                
                fig_qq = px.scatter(
                    x=qq_data[0][0],
                    y=qq_data[0][1],
                    title="Q-Q Plot of Residuals"
                )
                st.plotly_chart(fig_qq, use_container_width=True)
    
    # Feature Analysis
    st.subheader("📊 Feature Analysis")
    
    # Feature statistics
    feature_stats = model_explainer.get_feature_statistics(X)
    
    if not feature_stats.empty:
        st.subheader("📈 Feature Statistics")
        st.dataframe(feature_stats, use_container_width=True)
    
    # Feature correlations
    st.subheader("📊 Feature Correlations")
    
    corr_data = model_explainer.get_feature_correlation_matrix(X)
    
    if 'error' not in corr_data:
        # Create correlation heatmap
        import plotly.express as px
        
        corr_matrix = pd.DataFrame(
            corr_data['correlation_matrix'],
            index=corr_data['features'],
            columns=corr_data['features']
        )
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Partial Dependence
    st.subheader("📊 Partial Dependence Analysis")
    
    # Select feature for partial dependence
    if not feature_cols:
        st.warning("No features available for partial dependence analysis.")
    else:
        selected_feature = st.selectbox("Select Feature for Partial Dependence", feature_cols)
        
        if st.button("📊 Calculate Partial Dependence"):
            try:
                pd_data = model_explainer.get_partial_dependence_data(
                    model, X, selected_feature, model_name
                )
                
                if 'error' not in pd_data:
                    # Create partial dependence plot
                    import plotly.express as px
                    
                    pd_df = pd.DataFrame({
                        'Feature Value': pd_data['feature_values'],
                        'Partial Dependence': pd_data['partial_dependence']
                    })
                    
                    fig_pd = px.line(
                        pd_df,
                        x='Feature Value',
                        y='Partial Dependence',
                        title=f"Partial Dependence - {selected_feature}"
                    )
                    st.plotly_chart(fig_pd, use_container_width=True)
                else:
                    st.error(f"Partial dependence error: {pd_data['error']}")
                    
            except Exception as e:
                st.error(f"Partial dependence error: {str(e)}")
    
    # Export Results
    st.subheader("💾 Export Results")
    
    if 'explanations' in st.session_state:
        if st.button("📥 Download Explainability Results"):
            explanations = st.session_state.explanations
            
            # Export feature importance
            feature_importance = explanations.get('feature_importance', pd.DataFrame())
            
            if not feature_importance.empty:
                csv = feature_importance.to_csv(index=False)
                st.download_button(
                    label="Download Feature Importance CSV",
                    data=csv,
                    file_name=f"feature_importance_{selected_symbol}_{selected_horizon}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
