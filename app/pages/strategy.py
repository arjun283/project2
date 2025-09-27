"""
Strategy page for the Commodity Price Predictor app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.data_loader import DataLoader
from services.features import FeatureEngineer
from services.strategy import StrategySimulator
from services.plots import PlotBuilder
from utils.constants import COMMODITIES, HORIZONS

def show_strategy():
    """Display the strategy page."""
    
    st.title("⚡ Strategy Simulator")
    st.markdown("Create and simulate trading strategies with advanced risk management.")
    
    # Get services from session state
    data_loader = st.session_state.data_loader
    feature_engineer = st.session_state.feature_engineer
    strategy_simulator = st.session_state.strategy_simulator
    plot_builder = st.session_state.plot_builder
    
    # Sidebar controls
    st.sidebar.subheader("Strategy Configuration")
    
    # Symbol selection
    symbols = list(COMMODITIES.keys())
    selected_symbol = st.sidebar.selectbox("Select Commodity", symbols)
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    
    strategy_name = st.sidebar.text_input("Strategy Name", value=f"{selected_symbol}_strategy")
    horizon = st.sidebar.selectbox("Prediction Horizon", list(HORIZONS.keys()))
    threshold = st.sidebar.slider("Signal Threshold", 0.001, 0.05, 0.01, 0.001)
    model_name = st.sidebar.selectbox("Model", ["naive_last", "naive_seasonal", "arima", "xgboost"])
    
    # Risk management
    st.sidebar.subheader("Risk Management")
    
    max_position = st.sidebar.slider("Max Position Size", 0.1, 2.0, 1.0, 0.1)
    stop_loss = st.sidebar.slider("Stop Loss (%)", 0.0, 0.2, 0.0, 0.01)
    take_profit = st.sidebar.slider("Take Profit (%)", 0.0, 0.2, 0.0, 0.01)
    
    # Transaction costs
    transaction_cost_bps = st.sidebar.slider(
        "Transaction Cost (bps)",
        0, 50, 5, 1
    )
    
    # Initial capital
    initial_capital = st.sidebar.number_input(
        "Initial Capital",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000
    )
    
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
    
    # Create Strategy
    st.subheader("🚀 Create Strategy")
    
    if st.button("⚡ Create Strategy", type="primary"):
        try:
            # Create strategy
            strategy = strategy_simulator.create_strategy(
                name=strategy_name,
                symbol=selected_symbol,
                horizon=horizon,
                threshold=threshold,
                model_name=model_name,
                max_position=max_position,
                stop_loss=stop_loss if stop_loss > 0 else None,
                take_profit=take_profit if take_profit > 0 else None
            )
            
            st.success(f"Strategy '{strategy_name}' created successfully!")
            
        except Exception as e:
            st.error(f"Error creating strategy: {str(e)}")
    
    # Simulate Strategy
    st.subheader("📊 Simulate Strategy")
    
    if st.button("🎯 Simulate Strategy", type="primary"):
        try:
            # Create features
            df_with_features = feature_engineer.create_features(df, selected_symbol)
            df_with_features = feature_engineer.create_targets(df_with_features, [horizon])
            
            # Simulate strategy
            result = strategy_simulator.simulate_strategy(
                strategy_name, df_with_features, initial_capital
            )
            
            if 'error' not in result:
                st.session_state.strategy_result = result
                st.success("Strategy simulation completed!")
            else:
                st.error(f"Simulation error: {result['error']}")
                
        except Exception as e:
            st.error(f"Simulation error: {str(e)}")
    
    # Display Results
    if 'strategy_result' in st.session_state:
        st.subheader("📊 Strategy Results")
        
        result = st.session_state.strategy_result
        
        if 'error' in result:
            st.error(f"Error: {result['error']}")
            return
        
        # Performance metrics
        metrics = result.get('metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
        with col2:
            st.metric("Annualized Return", f"{metrics.get('annualized_return', 0):.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        with col4:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        
        # Performance chart
        st.subheader("📈 Performance Chart")
        
        returns = result.get('returns', pd.Series())
        if not returns.empty:
            fig = plot_builder.create_performance_chart(
                returns, 
                f"{strategy_name} Performance",
                initial_capital
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade analysis
        st.subheader("📊 Trade Analysis")
        
        trade_stats = result.get('trade_stats', {})
        
        if 'error' not in trade_stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", trade_stats.get('n_trades', 0))
                st.metric("Win Rate", f"{trade_stats.get('win_rate', 0):.1%}")
            
            with col2:
                st.metric("Avg Trade Return", f"{trade_stats.get('avg_trade_return', 0):.4f}")
                st.metric("Best Trade", f"{trade_stats.get('best_trade', 0):.4f}")
            
            with col3:
                st.metric("Worst Trade", f"{trade_stats.get('worst_trade', 0):.4f}")
                st.metric("Avg Trade Duration", f"{trade_stats.get('avg_trade_duration', 0):.1f} days")
        
        # Signals analysis
        st.subheader("📊 Signals Analysis")
        
        signals = result.get('signals', pd.Series())
        positions = result.get('positions', pd.Series())
        
        if not signals.empty:
            # Signal distribution
            signal_counts = signals.value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Signal Distribution")
                signal_data = pd.DataFrame({
                    'Signal': ['Long', 'Flat', 'Short'],
                    'Count': [
                        signal_counts.get(1, 0),
                        signal_counts.get(0, 0),
                        signal_counts.get(-1, 0)
                    ]
                })
                
                import plotly.express as px
                fig_signals = px.bar(
                    signal_data,
                    x='Signal',
                    y='Count',
                    title="Signal Distribution"
                )
                st.plotly_chart(fig_signals, use_container_width=True)
            
            with col2:
                st.subheader("Position Distribution")
                position_data = pd.DataFrame({
                    'Position': ['Long', 'Flat', 'Short'],
                    'Count': [
                        (positions > 0).sum(),
                        (positions == 0).sum(),
                        (positions < 0).sum()
                    ]
                })
                
                fig_positions = px.bar(
                    position_data,
                    x='Position',
                    y='Count',
                    title="Position Distribution"
                )
                st.plotly_chart(fig_positions, use_container_width=True)
    
    # Portfolio Management
    st.subheader("💼 Portfolio Management")
    
    # Create portfolio
    st.subheader("Create Portfolio")
    
    portfolio_name = st.text_input("Portfolio Name", value="my_portfolio")
    
    # Get available strategies
    strategy_summary = strategy_simulator.get_strategy_summary()
    
    if not strategy_summary.empty:
        available_strategies = strategy_summary['name'].tolist()
        
        selected_strategies = st.multiselect(
            "Select Strategies",
            available_strategies,
            default=available_strategies[:2] if len(available_strategies) >= 2 else available_strategies
        )
        
        if selected_strategies:
            # Portfolio weights
            st.subheader("Portfolio Weights")
            
            weights = {}
            for strategy in selected_strategies:
                weight = st.number_input(
                    f"Weight for {strategy}",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0 / len(selected_strategies),
                    step=0.01
                )
                weights[strategy] = weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            if st.button("💼 Create Portfolio"):
                try:
                    portfolio = strategy_simulator.create_portfolio(
                        name=portfolio_name,
                        strategies=selected_strategies,
                        weights=weights
                    )
                    
                    st.success(f"Portfolio '{portfolio_name}' created successfully!")
                    
                except Exception as e:
                    st.error(f"Error creating portfolio: {str(e)}")
    
    # Display Portfolio Summary
    st.subheader("📊 Portfolio Summary")
    
    portfolio_summary = strategy_simulator.get_portfolio_summary()
    
    if not portfolio_summary.empty:
        st.dataframe(portfolio_summary, use_container_width=True)
    else:
        st.info("No portfolios created yet.")
    
    # Export Results
    st.subheader("💾 Export Results")
    
    if 'strategy_result' in st.session_state:
        if st.button("📥 Download Strategy Results"):
            result = st.session_state.strategy_result
            
            # Create export data
            export_data = {
                'strategy_name': result.get('strategy', {}).get('name', ''),
                'symbol': result.get('strategy', {}).get('symbol', ''),
                'horizon': result.get('strategy', {}).get('horizon', ''),
                'threshold': result.get('strategy', {}).get('threshold', 0),
                'model': result.get('strategy', {}).get('model_name', ''),
                'initial_capital': result.get('initial_capital', 0),
                'total_return': result.get('metrics', {}).get('total_return', 0),
                'annualized_return': result.get('metrics', {}).get('annualized_return', 0),
                'sharpe_ratio': result.get('metrics', {}).get('sharpe_ratio', 0),
                'max_drawdown': result.get('metrics', {}).get('max_drawdown', 0),
                'hit_rate': result.get('metrics', {}).get('hit_rate', 0),
                'n_trades': result.get('metrics', {}).get('n_trades', 0)
            }
            
            export_df = pd.DataFrame([export_data])
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name=f"strategy_results_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
