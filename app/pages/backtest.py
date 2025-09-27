"""
Backtest page for the Commodity Price Predictor app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.data_loader import DataLoader
from services.features import FeatureEngineer
from services.backtest import BacktestEngine
from services.plots import PlotBuilder
from utils.constants import COMMODITIES, HORIZONS

def show_backtest():
    """Display the backtest page."""
    
    st.title("📈 Backtesting")
    st.markdown("Run backtests on commodity trading strategies with realistic transaction costs.")
    
    # Get services from session state
    data_loader = st.session_state.data_loader
    feature_engineer = st.session_state.feature_engineer
    backtest_engine = st.session_state.backtest_engine
    plot_builder = st.session_state.plot_builder
    
    # Sidebar controls
    st.sidebar.subheader("Backtest Configuration")
    
    # Symbol selection
    symbols = list(COMMODITIES.keys())
    selected_symbols = st.sidebar.multiselect(
        "Select Commodities",
        symbols,
        default=symbols[:3]
    )
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    
    horizon = st.sidebar.selectbox("Prediction Horizon", list(HORIZONS.keys()))
    threshold = st.sidebar.slider("Signal Threshold", 0.001, 0.05, 0.01, 0.001)
    model_name = st.sidebar.selectbox("Model", ["naive_last", "naive_seasonal", "arima", "xgboost"])
    
    # Transaction costs
    transaction_cost_bps = st.sidebar.slider(
        "Transaction Cost (bps)",
        0, 50, 5, 1
    )
    
    # Date range
    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*2))
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
    
    # Check if data is loaded
    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data first using the sidebar controls.")
        return
    
    data = st.session_state.data
    
    # Run Backtest
    st.subheader("🚀 Run Backtest")
    
    if st.button("📊 Run Backtest", type="primary"):
        if not selected_symbols:
            st.error("Please select commodities to backtest.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        backtest_results = {}
        total_symbols = len(selected_symbols)
        
        for i, symbol in enumerate(selected_symbols):
            if symbol not in data or data[symbol].empty:
                continue
            
            progress_bar.progress((i + 1) / total_symbols)
            status_text.text(f"Backtesting {symbol}...")
            
            # Create features
            df = feature_engineer.create_features(data[symbol], symbol)
            df = feature_engineer.create_targets(df, [horizon])
            
            # Strategy configuration
            strategy_config = {
                'horizon': horizon,
                'threshold': threshold,
                'model': model_name
            }
            
            # Run backtest
            try:
                result = backtest_engine.run_backtest(
                    df, symbol, strategy_config,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if 'error' not in result:
                    backtest_results[symbol] = result
                else:
                    st.error(f"Error backtesting {symbol}: {result['error']}")
                    
            except Exception as e:
                st.error(f"Error backtesting {symbol}: {str(e)}")
        
        st.session_state.backtest_results = backtest_results
        progress_bar.progress(1.0)
        status_text.text("Backtest completed!")
        st.success("Backtest completed successfully!")
    
    # Display Results
    if 'backtest_results' in st.session_state:
        st.subheader("📊 Backtest Results")
        
        backtest_results = st.session_state.backtest_results
        
        if not backtest_results:
            st.warning("No backtest results available.")
            return
        
        # Summary Table
        st.subheader("📋 Performance Summary")
        
        summary_data = []
        for symbol, result in backtest_results.items():
            metrics = result.get('metrics', {})
            if 'error' in metrics:
                continue
            
            summary_data.append({
                'Symbol': symbol,
                'Total Return': f"{metrics.get('total_return', 0):.2%}",
                'Annualized Return': f"{metrics.get('annualized_return', 0):.2%}",
                'Volatility': f"{metrics.get('volatility', 0):.2%}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                'Hit Rate': f"{metrics.get('hit_rate', 0):.1%}",
                'N Trades': metrics.get('n_trades', 0)
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Best performing strategy
            if len(summary_data) > 1:
                best_symbol = summary_df.loc[summary_df['Sharpe Ratio'].astype(float).idxmax(), 'Symbol']
                st.success(f"🏆 Best performing strategy: **{best_symbol}**")
        
        # Individual Strategy Analysis
        st.subheader("🔍 Individual Strategy Analysis")
        
        selected_symbol = st.selectbox("Select Strategy", list(backtest_results.keys()))
        
        if selected_symbol in backtest_results:
            result = backtest_results[selected_symbol]
            
            if 'error' in result:
                st.error(f"Error in {selected_symbol}: {result['error']}")
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
                # Calculate equity curve
                initial_capital = 10000
                equity_curve = backtest_engine.get_equity_curve(returns, initial_capital)
                
                # Create performance chart
                fig = plot_builder.create_performance_chart(returns, f"{selected_symbol} Strategy Performance")
                st.plotly_chart(fig, use_container_width=True)
                
                # Rolling metrics
                st.subheader("📊 Rolling Metrics")
                
                rolling_metrics = backtest_engine.get_rolling_metrics(returns, window=252)
                
                if not rolling_metrics.empty:
                    # Rolling Sharpe ratio
                    import plotly.express as px
                    fig_rolling = px.line(
                        rolling_metrics.reset_index(),
                        x='Date',
                        y='rolling_sharpe',
                        title="Rolling Sharpe Ratio (252 days)"
                    )
                    st.plotly_chart(fig_rolling, use_container_width=True)
                
                # Trade analysis
                st.subheader("📊 Trade Analysis")
                
                trade_stats = {
                    'Total Trades': metrics.get('n_trades', 0),
                    'Winning Trades': metrics.get('n_winning_trades', 0),
                    'Losing Trades': metrics.get('n_losing_trades', 0),
                    'Win Rate': f"{metrics.get('win_rate', 0):.1%}",
                    'Avg Win': f"{metrics.get('avg_win', 0):.4f}",
                    'Avg Loss': f"{metrics.get('avg_loss', 0):.4f}",
                    'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}"
                }
                
                col1, col2, col3 = st.columns(3)
                for i, (key, value) in enumerate(trade_stats.items()):
                    with [col1, col2, col3][i % 3]:
                        st.metric(key, value)
        
        # Portfolio Backtest
        if len(backtest_results) > 1:
            st.subheader("💼 Portfolio Backtest")
            
            # Portfolio configuration
            col1, col2 = st.columns(2)
            
            with col1:
                portfolio_type = st.selectbox(
                    "Portfolio Type",
                    ["Equal Weight", "Custom Weights"]
                )
            
            with col2:
                if portfolio_type == "Custom Weights":
                    weights = {}
                    for symbol in backtest_results.keys():
                        weight = st.number_input(
                            f"Weight for {symbol}",
                            min_value=0.0,
                            max_value=1.0,
                            value=1.0 / len(backtest_results),
                            step=0.01
                        )
                        weights[symbol] = weight
                else:
                    weights = {symbol: 1.0 / len(backtest_results) for symbol in backtest_results.keys()}
            
            if st.button("📊 Run Portfolio Backtest"):
                # Prepare individual results
                individual_results = {}
                for symbol, result in backtest_results.items():
                    if 'returns' in result:
                        individual_results[symbol] = result
                
                # Portfolio configuration
                portfolio_config = {
                    'strategy': {
                        'horizon': horizon,
                        'threshold': threshold,
                        'model': model_name
                    },
                    'weights': weights
                }
                
                # Run portfolio backtest
                try:
                    portfolio_result = backtest_engine.run_portfolio_backtest(
                        individual_results,
                        portfolio_config,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if 'error' not in portfolio_result:
                        st.session_state.portfolio_result = portfolio_result
                        st.success("Portfolio backtest completed!")
                    else:
                        st.error(f"Portfolio backtest error: {portfolio_result['error']}")
                        
                except Exception as e:
                    st.error(f"Portfolio backtest error: {str(e)}")
        
        # Display Portfolio Results
        if 'portfolio_result' in st.session_state:
            st.subheader("💼 Portfolio Results")
            
            portfolio_result = st.session_state.portfolio_result
            portfolio_metrics = portfolio_result.get('portfolio_metrics', {})
            
            if 'error' not in portfolio_metrics:
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
                portfolio_returns = portfolio_result.get('portfolio_returns', pd.Series())
                if not portfolio_returns.empty:
                    fig = plot_builder.create_performance_chart(
                        portfolio_returns, 
                        "Portfolio Performance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Export Results
    st.subheader("💾 Export Results")
    
    if 'backtest_results' in st.session_state:
        if st.button("📥 Download Backtest Results"):
            # Create summary for export
            export_data = []
            for symbol, result in st.session_state.backtest_results.items():
                metrics = result.get('metrics', {})
                if 'error' in metrics:
                    continue
                
                export_data.append({
                    'Symbol': symbol,
                    'Horizon': horizon,
                    'Threshold': threshold,
                    'Model': model_name,
                    'Total_Return': metrics.get('total_return', 0),
                    'Annualized_Return': metrics.get('annualized_return', 0),
                    'Volatility': metrics.get('volatility', 0),
                    'Sharpe_Ratio': metrics.get('sharpe_ratio', 0),
                    'Max_Drawdown': metrics.get('max_drawdown', 0),
                    'Hit_Rate': metrics.get('hit_rate', 0),
                    'N_Trades': metrics.get('n_trades', 0)
                })
            
            if export_data:
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
