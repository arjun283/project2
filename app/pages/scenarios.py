"""
Scenarios page for the Commodity Price Predictor app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.data_loader import DataLoader
from services.features import FeatureEngineer
from services.modeling import ModelTrainer
from services.plots import PlotBuilder
from utils.constants import COMMODITIES, HORIZONS

def show_scenarios():
    """Display the scenarios page."""
    
    st.title("🎯 Scenario Analysis")
    st.markdown("Analyze how different market scenarios affect commodity price predictions.")
    
    # Get services from session state
    data_loader = st.session_state.data_loader
    feature_engineer = st.session_state.feature_engineer
    model_trainer = st.session_state.model_trainer
    plot_builder = st.session_state.plot_builder
    
    # Sidebar controls
    st.sidebar.subheader("Scenario Configuration")
    
    # Symbol selection
    symbols = list(COMMODITIES.keys())
    selected_symbol = st.sidebar.selectbox("Select Commodity", symbols)
    
    # Horizon selection
    selected_horizon = st.sidebar.selectbox("Select Horizon", list(HORIZONS.keys()))
    
    # Model selection
    model_name = st.sidebar.selectbox("Select Model", ["xgboost", "arima", "naive_last"])
    
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
    
    # Train model
    st.subheader("🤖 Train Model for Scenario Analysis")
    
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
    
    # Get model
    models = horizon_results.get('models', {})
    if model_name not in models or 'error' in models[model_name]:
        st.error(f"Model {model_name} not available or has errors.")
        return
    
    model_data = models[model_name]
    if 'model' not in model_data:
        st.error(f"Model {model_name} not properly trained.")
        return
    
    model = model_data['model']
    
    # Scenario Parameters
    st.subheader("🎯 Scenario Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # USD strength scenario
        st.subheader("💵 USD Strength Scenario")
        usd_move_pct = st.slider(
            "USD Move (%)",
            -20.0, 20.0, 0.0, 0.5,
            help="Percentage change in USD strength"
        )
        
        # Interest rate scenario
        st.subheader("📈 Interest Rate Scenario")
        yield_change_bps = st.slider(
            "10Y Yield Change (bps)",
            -200, 200, 0, 10,
            help="Change in 10-year Treasury yield in basis points"
        )
    
    with col2:
        # Commodity-specific shock
        st.subheader("⚡ Commodity Shock Scenario")
        commodity_shock_pct = st.slider(
            "Commodity Shock (%)",
            -50.0, 50.0, 0.0, 1.0,
            help="Percentage shock to commodity price"
        )
        
        # Volatility scenario
        st.subheader("📊 Volatility Scenario")
        vol_multiplier = st.slider(
            "Volatility Multiplier",
            0.5, 3.0, 1.0, 0.1,
            help="Multiplier for current volatility"
        )
    
    # Run Scenario Analysis
    st.subheader("🚀 Run Scenario Analysis")
    
    if st.button("🎯 Run Scenarios", type="primary"):
        try:
            # Prepare base data
            feature_cols = [col for col in df_with_features.columns if col not in [
                'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'data_quality',
                'target_1d', 'target_5d', 'target_20d', 'price_1d', 'price_5d', 'price_20d',
                'direction_1d', 'direction_5d', 'direction_20d', 'regime', 'regime_change'
            ]]
            
            X_base = df_with_features[feature_cols].fillna(0)
            
            # Create scenario data
            scenarios = {
                'Base Case': X_base.copy(),
                'USD Strong (+5%)': X_base.copy(),
                'USD Weak (-5%)': X_base.copy(),
                'Rates Up (+100bps)': X_base.copy(),
                'Rates Down (-100bps)': X_base.copy(),
                'Commodity Shock (+10%)': X_base.copy(),
                'Commodity Shock (-10%)': X_base.copy(),
                'High Volatility': X_base.copy(),
                'Custom Scenario': X_base.copy()
            }
            
            # Apply scenario modifications
            # USD strength affects cross-asset features
            if 'UUP_returns' in X_base.columns:
                scenarios['USD Strong (+5%)']['UUP_returns'] *= 1.05
                scenarios['USD Weak (-5%)']['UUP_returns'] *= 0.95
            
            # Interest rate changes affect yield features
            if 'DGS10_returns' in X_base.columns:
                scenarios['Rates Up (+100bps)']['DGS10_returns'] += 0.01
                scenarios['Rates Down (-100bps)']['DGS10_returns'] -= 0.01
            
            # Commodity shocks affect price features
            price_features = [col for col in X_base.columns if 'price' in col.lower() or 'close' in col.lower()]
            for feature in price_features:
                if feature in X_base.columns:
                    scenarios['Commodity Shock (+10%)'][feature] *= 1.10
                    scenarios['Commodity Shock (-10%)'][feature] *= 0.90
            
            # Volatility changes
            vol_features = [col for col in X_base.columns if 'vol' in col.lower()]
            for feature in vol_features:
                if feature in X_base.columns:
                    scenarios['High Volatility'][feature] *= 2.0
            
            # Custom scenario
            if usd_move_pct != 0 and 'UUP_returns' in X_base.columns:
                scenarios['Custom Scenario']['UUP_returns'] *= (1 + usd_move_pct / 100)
            
            if yield_change_bps != 0 and 'DGS10_returns' in X_base.columns:
                scenarios['Custom Scenario']['DGS10_returns'] += yield_change_bps / 10000
            
            if commodity_shock_pct != 0:
                for feature in price_features:
                    if feature in X_base.columns:
                        scenarios['Custom Scenario'][feature] *= (1 + commodity_shock_pct / 100)
            
            if vol_multiplier != 1.0:
                for feature in vol_features:
                    if feature in X_base.columns:
                        scenarios['Custom Scenario'][feature] *= vol_multiplier
            
            # Make predictions for each scenario
            scenario_predictions = {}
            
            for scenario_name, X_scenario in scenarios.items():
                try:
                    # Make predictions
                    if hasattr(model, 'predict'):
                        predictions = model.predict(X_scenario)
                    else:
                        predictions = np.full(len(X_scenario), 0)
                    
                    scenario_predictions[scenario_name] = {
                        'predictions': predictions,
                        'mean_prediction': np.mean(predictions),
                        'std_prediction': np.std(predictions),
                        'positive_pct': np.mean(predictions > 0) * 100
                    }
                    
                except Exception as e:
                    st.error(f"Error in scenario {scenario_name}: {str(e)}")
                    scenario_predictions[scenario_name] = {
                        'predictions': np.full(len(X_scenario), 0),
                        'mean_prediction': 0,
                        'std_prediction': 0,
                        'positive_pct': 0
                    }
            
            st.session_state.scenario_predictions = scenario_predictions
            st.success("Scenario analysis completed!")
            
        except Exception as e:
            st.error(f"Scenario analysis error: {str(e)}")
    
    # Display Scenario Results
    if 'scenario_predictions' in st.session_state:
        st.subheader("📊 Scenario Analysis Results")
        
        scenario_predictions = st.session_state.scenario_predictions
        
        # Scenario comparison table
        st.subheader("📋 Scenario Comparison")
        
        comparison_data = []
        for scenario_name, results in scenario_predictions.items():
            comparison_data.append({
                'Scenario': scenario_name,
                'Mean Prediction': f"{results['mean_prediction']:.4f}",
                'Std Prediction': f"{results['std_prediction']:.4f}",
                'Positive %': f"{results['positive_pct']:.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Scenario comparison chart
        st.subheader("📊 Scenario Comparison Chart")
        
        import plotly.express as px
        
        # Mean predictions chart
        fig_mean = px.bar(
            comparison_df,
            x='Scenario',
            y='Mean Prediction',
            title=f"Mean Predictions by Scenario - {selected_symbol} {selected_horizon}"
        )
        st.plotly_chart(fig_mean, use_container_width=True)
        
        # Positive percentage chart
        fig_positive = px.bar(
            comparison_df,
            x='Scenario',
            y='Positive %',
            title=f"Positive Prediction % by Scenario - {selected_symbol} {selected_horizon}"
        )
        st.plotly_chart(fig_positive, use_container_width=True)
        
        # Sensitivity analysis
        st.subheader("📊 Sensitivity Analysis")
        
        # Calculate sensitivity to each factor
        base_mean = scenario_predictions['Base Case']['mean_prediction']
        
        sensitivity_data = []
        
        # USD sensitivity
        usd_strong_mean = scenario_predictions['USD Strong (+5%)']['mean_prediction']
        usd_weak_mean = scenario_predictions['USD Weak (-5%)']['mean_prediction']
        usd_sensitivity = (usd_strong_mean - usd_weak_mean) / 10  # Per 1% change
        
        sensitivity_data.append({
            'Factor': 'USD Strength',
            'Sensitivity': usd_sensitivity,
            'Base Case': base_mean,
            'Strong Case': usd_strong_mean,
            'Weak Case': usd_weak_mean
        })
        
        # Interest rate sensitivity
        rates_up_mean = scenario_predictions['Rates Up (+100bps)']['mean_prediction']
        rates_down_mean = scenario_predictions['Rates Down (-100bps)']['mean_prediction']
        rates_sensitivity = (rates_up_mean - rates_down_mean) / 200  # Per 1bp change
        
        sensitivity_data.append({
            'Factor': 'Interest Rates',
            'Sensitivity': rates_sensitivity,
            'Base Case': base_mean,
            'Up Case': rates_up_mean,
            'Down Case': rates_down_mean
        })
        
        # Commodity shock sensitivity
        shock_up_mean = scenario_predictions['Commodity Shock (+10%)']['mean_prediction']
        shock_down_mean = scenario_predictions['Commodity Shock (-10%)']['mean_prediction']
        shock_sensitivity = (shock_up_mean - shock_down_mean) / 20  # Per 1% change
        
        sensitivity_data.append({
            'Factor': 'Commodity Shock',
            'Sensitivity': shock_sensitivity,
            'Base Case': base_mean,
            'Up Case': shock_up_mean,
            'Down Case': shock_down_mean
        })
        
        sensitivity_df = pd.DataFrame(sensitivity_data)
        
        # Display sensitivity table
        st.dataframe(sensitivity_df, use_container_width=True)
        
        # Sensitivity chart
        fig_sensitivity = px.bar(
            sensitivity_df,
            x='Factor',
            y='Sensitivity',
            title="Factor Sensitivity Analysis"
        )
        st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        # Tornado chart
        st.subheader("🌪️ Tornado Chart")
        
        tornado_data = []
        for _, row in sensitivity_df.iterrows():
            tornado_data.append({
                'Factor': row['Factor'],
                'Impact': abs(row['Sensitivity']),
                'Direction': 'Positive' if row['Sensitivity'] > 0 else 'Negative'
            })
        
        tornado_df = pd.DataFrame(tornado_data)
        tornado_df = tornado_df.sort_values('Impact', ascending=True)
        
        fig_tornado = px.bar(
            tornado_df,
            x='Impact',
            y='Factor',
            orientation='h',
            color='Direction',
            title="Factor Impact (Tornado Chart)"
        )
        st.plotly_chart(fig_tornado, use_container_width=True)
        
        # Custom scenario analysis
        if 'Custom Scenario' in scenario_predictions:
            st.subheader("🎯 Custom Scenario Analysis")
            
            custom_results = scenario_predictions['Custom Scenario']
            base_results = scenario_predictions['Base Case']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Mean Prediction",
                    f"{custom_results['mean_prediction']:.4f}",
                    f"{custom_results['mean_prediction'] - base_results['mean_prediction']:.4f}"
                )
            
            with col2:
                st.metric(
                    "Std Prediction",
                    f"{custom_results['std_prediction']:.4f}",
                    f"{custom_results['std_prediction'] - base_results['std_prediction']:.4f}"
                )
            
            with col3:
                st.metric(
                    "Positive %",
                    f"{custom_results['positive_pct']:.1f}%",
                    f"{custom_results['positive_pct'] - base_results['positive_pct']:.1f}%"
                )
            
            with col4:
                change_pct = ((custom_results['mean_prediction'] - base_results['mean_prediction']) / 
                            abs(base_results['mean_prediction']) * 100) if base_results['mean_prediction'] != 0 else 0
                st.metric("Change %", f"{change_pct:.1f}%")
    
    # Export Results
    st.subheader("💾 Export Results")
    
    if 'scenario_predictions' in st.session_state:
        if st.button("📥 Download Scenario Results"):
            scenario_predictions = st.session_state.scenario_predictions
            
            # Create export data
            export_data = []
            for scenario_name, results in scenario_predictions.items():
                export_data.append({
                    'Scenario': scenario_name,
                    'Mean_Prediction': results['mean_prediction'],
                    'Std_Prediction': results['std_prediction'],
                    'Positive_Pct': results['positive_pct']
                })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Scenario Results CSV",
                data=csv,
                file_name=f"scenario_results_{selected_symbol}_{selected_horizon}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
