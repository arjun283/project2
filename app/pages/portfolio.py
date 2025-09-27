"""
Portfolio page for the Commodity Price Predictor app.
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

def show_portfolio():
    """Display the portfolio page."""
    
    st.title("💼 Portfolio Management")
    st.markdown("Manage and analyze portfolios of commodity trading strategies.")
    
    # Get services from session state
    data_loader = st.session_state.data_loader
    feature_engineer = st.session_state.feature_engineer
    strategy_simulator = st.session_state.strategy_simulator
    plot_builder = st.session_state.plot_builder
    
    # Sidebar controls
    st.sidebar.subheader("Portfolio Configuration")
    
    # Symbol selection
    symbols = list(COMMODITIES.keys())
    selected_symbols = st.sidebar.multiselect(
        "Select Commodities",
        symbols,
        default=symbols[:3]
    )
    
    # Portfolio parameters
    st.sidebar.subheader("Portfolio Parameters")
    
    portfolio_name = st.sidebar.text_input("Portfolio Name", value="commodity_portfolio")
    rebalance_frequency = st.sidebar.selectbox(
        "Rebalance Frequency",
        ["daily", "weekly", "monthly", "quarterly"]
    )
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    
    horizon = st.sidebar.selectbox("Prediction Horizon", list(HORIZONS.keys()))
    threshold = st.sidebar.slider("Signal Threshold", 0.001, 0.05, 0.01, 0.001)
    model_name = st.sidebar.selectbox("Model", ["naive_last", "naive_seasonal", "arima", "xgboost"])
    
    # Risk management
    st.sidebar.subheader("Risk Management")
    
    max_position = st.sidebar.slider("Max Position Size", 0.1, 2.0, 1.0, 0.1)
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
            data = data_loader.load_commodity_data(selected_symbols)
            st.session_state.data = data
            st.success(f"Loaded data for {len(data)} commodities")
    
    # Check if data is loaded
    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data first using the sidebar controls.")
        return
    
    data = st.session_state.data
    
    # Create Individual Strategies
    st.subheader("🚀 Create Individual Strategies")
    
    if st.button("⚡ Create Strategies", type="primary"):
        if not selected_symbols:
            st.error("Please select commodities first.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        strategies_created = 0
        total_symbols = len(selected_symbols)
        
        for i, symbol in enumerate(selected_symbols):
            if symbol not in data or data[symbol].empty:
                continue
            
            progress_bar.progress((i + 1) / total_symbols)
            status_text.text(f"Creating strategy for {symbol}...")
            
            try:
                # Create features
                df = feature_engineer.create_features(data[symbol], symbol)
                df = feature_engineer.create_targets(df, [horizon])
                
                # Create strategy
                strategy_name = f"{symbol}_{horizon}_{model_name}"
                strategy = strategy_simulator.create_strategy(
                    name=strategy_name,
                    symbol=symbol,
                    horizon=horizon,
                    threshold=threshold,
                    model_name=model_name,
                    max_position=max_position
                )
                
                strategies_created += 1
                
            except Exception as e:
                st.error(f"Error creating strategy for {symbol}: {str(e)}")
        
        progress_bar.progress(1.0)
        status_text.text("Strategy creation completed!")
        st.success(f"Created {strategies_created} strategies successfully!")
    
    # Create Portfolio
    st.subheader("💼 Create Portfolio")
    
    # Get available strategies
    strategy_summary = strategy_simulator.get_strategy_summary()
    
    if not strategy_summary.empty:
        available_strategies = strategy_summary['name'].tolist()
        
        selected_strategies = st.multiselect(
            "Select Strategies for Portfolio",
            available_strategies,
            default=available_strategies[:3] if len(available_strategies) >= 3 else available_strategies
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
            
            if st.button("💼 Create Portfolio", type="primary"):
                try:
                    portfolio = strategy_simulator.create_portfolio(
                        name=portfolio_name,
                        strategies=selected_strategies,
                        weights=weights,
                        rebalance_frequency=rebalance_frequency
                    )
                    
                    st.success(f"Portfolio '{portfolio_name}' created successfully!")
                    
                except Exception as e:
                    st.error(f"Error creating portfolio: {str(e)}")
    
    # Simulate Portfolio
    st.subheader("📊 Simulate Portfolio")
    
    if st.button("🎯 Simulate Portfolio", type="primary"):
        if not selected_strategies:
            st.error("Please create strategies and portfolio first.")
            return
        
        try:
            # Prepare data for simulation
            simulation_data = {}
            for symbol in selected_symbols:
                if symbol in data and not data[symbol].empty:
                    df = feature_engineer.create_features(data[symbol], symbol)
                    df = feature_engineer.create_targets(df, [horizon])
                    simulation_data[symbol] = df
            
            # Simulate portfolio
            result = strategy_simulator.simulate_portfolio(
                portfolio_name, simulation_data, initial_capital
            )
            
            if 'error' not in result:
                st.session_state.portfolio_result = result
                st.success("Portfolio simulation completed!")
            else:
                st.error(f"Simulation error: {result['error']}")
                
        except Exception as e:
            st.error(f"Simulation error: {str(e)}")
    
    # Display Portfolio Results
    if 'portfolio_result' in st.session_state:
        st.subheader("📊 Portfolio Results")
        
        result = st.session_state.portfolio_result
        
        if 'error' in result:
            st.error(f"Error: {result['error']}")
            return
        
        # Portfolio metrics
        portfolio_metrics = result.get('portfolio_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Portfolio Return", f"{portfolio_metrics.get('total_return', 0):.2%}")
        with col2:
            st.metric("Portfolio Sharpe", f"{portfolio_metrics.get('sharpe_ratio', 0):.2f}")
        with col3:
            st.metric("Portfolio Volatility", f"{portfolio_metrics.get('volatility', 0):.2%}")
        with col4:
            st.metric("Portfolio Max DD", f"{portfolio_metrics.get('max_drawdown', 0):.2%}")
        
        # Portfolio performance chart
        st.subheader("📈 Portfolio Performance")
        
        portfolio_returns = result.get('portfolio_returns', pd.Series())
        if not portfolio_returns.empty:
            fig = plot_builder.create_performance_chart(
                portfolio_returns, 
                f"{portfolio_name} Portfolio Performance",
                initial_capital
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual strategy performance
        st.subheader("📊 Individual Strategy Performance")
        
        strategy_results = result.get('strategy_results', {})
        
        if strategy_results:
            strategy_summary_data = []
            for strategy_name, strategy_result in strategy_results.items():
                if 'error' in strategy_result:
                    continue
                
                metrics = strategy_result.get('metrics', {})
                strategy_summary_data.append({
                    'Strategy': strategy_name,
                    'Total Return': f"{metrics.get('total_return', 0):.2%}",
                    'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                    'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                    'Hit Rate': f"{metrics.get('hit_rate', 0):.1%}",
                    'N Trades': metrics.get('n_trades', 0)
                })
            
            if strategy_summary_data:
                strategy_summary_df = pd.DataFrame(strategy_summary_data)
                st.dataframe(strategy_summary_df, use_container_width=True)
                
                # Strategy comparison chart
                import plotly.express as px
                fig_comparison = px.bar(
                    strategy_summary_df,
                    x='Strategy',
                    y='Total Return',
                    title="Strategy Performance Comparison"
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Portfolio composition
        st.subheader("📊 Portfolio Composition")
        
        portfolio_info = result.get('portfolio', {})
        weights = portfolio_info.get('weights', {})
        
        if weights:
            # Create pie chart of portfolio weights
            import plotly.express as px
            fig_weights = px.pie(
                values=list(weights.values()),
                names=list(weights.keys()),
                title="Portfolio Weights"
            )
            st.plotly_chart(fig_weights, use_container_width=True)
            
            # Weights table
            weights_df = pd.DataFrame([
                {'Strategy': strategy, 'Weight': f"{weight:.1%}"}
                for strategy, weight in weights.items()
            ])
            st.dataframe(weights_df, use_container_width=True)
        
        # Risk metrics
        st.subheader("📊 Risk Metrics")
        
        risk_metrics = {
            'VaR (95%)': f"{portfolio_metrics.get('var_95', 0):.4f}",
            'CVaR (95%)': f"{portfolio_metrics.get('cvar_95', 0):.4f}",
            'Calmar Ratio': f"{portfolio_metrics.get('calmar_ratio', 0):.2f}",
            'Sortino Ratio': f"{portfolio_metrics.get('sortino_ratio', 0):.2f}",
            'Profit Factor': f"{portfolio_metrics.get('profit_factor', 0):.2f}"
        }
        
        col1, col2, col3 = st.columns(3)
        for i, (metric, value) in enumerate(risk_metrics.items()):
            with [col1, col2, col3][i % 3]:
                st.metric(metric, value)
    
    # Portfolio Summary
    st.subheader("📋 Portfolio Summary")
    
    portfolio_summary = strategy_simulator.get_portfolio_summary()
    
    if not portfolio_summary.empty:
        st.dataframe(portfolio_summary, use_container_width=True)
    else:
        st.info("No portfolios created yet.")
    
    # Export Results
    st.subheader("💾 Export Results")
    
    if 'portfolio_result' in st.session_state:
        if st.button("📥 Download Portfolio Results"):
            result = st.session_state.portfolio_result
            
            # Create export data
            portfolio_metrics = result.get('portfolio_metrics', {})
            export_data = {
                'portfolio_name': result.get('portfolio', {}).get('name', ''),
                'n_strategies': result.get('portfolio', {}).get('n_strategies', 0),
                'rebalance_frequency': result.get('portfolio', {}).get('rebalance_frequency', ''),
                'initial_capital': result.get('initial_capital', 0),
                'total_return': portfolio_metrics.get('total_return', 0),
                'annualized_return': portfolio_metrics.get('annualized_return', 0),
                'sharpe_ratio': portfolio_metrics.get('sharpe_ratio', 0),
                'volatility': portfolio_metrics.get('volatility', 0),
                'max_drawdown': portfolio_metrics.get('max_drawdown', 0),
                'hit_rate': portfolio_metrics.get('hit_rate', 0),
                'var_95': portfolio_metrics.get('var_95', 0),
                'cvar_95': portfolio_metrics.get('cvar_95', 0),
                'calmar_ratio': portfolio_metrics.get('calmar_ratio', 0)
            }
            
            export_df = pd.DataFrame([export_data])
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name=f"portfolio_results_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
