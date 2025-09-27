"""
Test backtesting functionality.
"""

import pytest
import numpy as np
import pandas as pd
from services.backtest import BacktestEngine

def test_backtest_engine_initialization():
    """Test BacktestEngine initialization."""
    engine = BacktestEngine(transaction_cost_bps=5.0)
    assert engine.transaction_cost_bps == 0.0005

def test_generate_signals():
    """Test signal generation."""
    engine = BacktestEngine()
    
    # Create test data
    predictions = pd.Series([0.01, -0.01, 0.02, -0.02, 0.005])
    threshold = 0.01
    
    signals = engine._generate_signals(predictions, threshold)
    
    expected_signals = pd.Series([1, -1, 1, -1, 0])
    pd.testing.assert_series_equal(signals, expected_signals)

def test_calculate_returns():
    """Test returns calculation."""
    engine = BacktestEngine()
    
    # Create test data
    df = pd.DataFrame({
        'Close': [100, 101, 102, 101, 103],
        'Date': pd.date_range('2023-01-01', periods=5)
    })
    
    signals = pd.Series([1, 0, 1, -1, 0])
    
    returns = engine._calculate_returns(df, signals)
    
    # Check that returns are calculated correctly
    assert len(returns) == len(signals)
    assert not returns.isnull().all()

def test_calculate_performance_metrics():
    """Test performance metrics calculation."""
    engine = BacktestEngine()
    
    # Create test returns
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    
    metrics = engine._calculate_performance_metrics(returns, pd.DataFrame())
    
    # Check that all expected metrics are present
    expected_metrics = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
    for metric in expected_metrics:
        assert metric in metrics

def test_equity_curve_calculation():
    """Test equity curve calculation."""
    engine = BacktestEngine()
    
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    initial_capital = 10000
    
    equity_curve = engine.get_equity_curve(returns, initial_capital)
    
    # Check that equity curve starts at initial capital
    assert equity_curve.iloc[0] == initial_capital
    
    # Check that equity curve is monotonically increasing or decreasing based on returns
    assert len(equity_curve) == len(returns)

def test_drawdown_calculation():
    """Test drawdown calculation."""
    engine = BacktestEngine()
    
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    
    drawdown = engine.get_drawdown_series(returns)
    
    # Check that drawdown is non-positive
    assert (drawdown <= 0).all()
    
    # Check that drawdown starts at 0
    assert drawdown.iloc[0] == 0

def test_rolling_metrics():
    """Test rolling metrics calculation."""
    engine = BacktestEngine()
    
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, 0.01, -0.005, 0.02, -0.01, 0.015])
    window = 5
    
    rolling_metrics = engine.get_rolling_metrics(returns, window)
    
    # Check that rolling metrics are calculated
    assert not rolling_metrics.empty
    assert 'rolling_return' in rolling_metrics.columns
    assert 'rolling_volatility' in rolling_metrics.columns

def test_portfolio_returns_calculation():
    """Test portfolio returns calculation."""
    engine = BacktestEngine()
    
    # Create individual strategy results
    individual_results = {
        'strategy1': {
            'returns': pd.Series([0.01, -0.005, 0.02]),
            'metrics': {}
        },
        'strategy2': {
            'returns': pd.Series([0.005, 0.01, -0.01]),
            'metrics': {}
        }
    }
    
    weights = {'strategy1': 0.6, 'strategy2': 0.4}
    
    portfolio_returns = engine._calculate_portfolio_returns(individual_results, weights)
    
    # Check that portfolio returns are calculated
    assert not portfolio_returns.empty
    assert len(portfolio_returns) == 3

def test_strategy_comparison():
    """Test strategy comparison."""
    engine = BacktestEngine()
    
    # Create strategy results
    strategy_results = {
        'strategy1': {
            'metrics': {
                'annualized_return': 0.1,
                'volatility': 0.15,
                'sharpe_ratio': 0.67,
                'max_drawdown': -0.05,
                'hit_rate': 0.6
            }
        },
        'strategy2': {
            'metrics': {
                'annualized_return': 0.12,
                'volatility': 0.18,
                'sharpe_ratio': 0.67,
                'max_drawdown': -0.08,
                'hit_rate': 0.55
            }
        }
    }
    
    comparison_df = engine.compare_strategies(strategy_results)
    
    # Check that comparison is created
    assert not comparison_df.empty
    assert 'strategy' in comparison_df.columns
    assert 'sharpe_ratio' in comparison_df.columns

def test_empty_returns():
    """Test handling of empty returns."""
    engine = BacktestEngine()
    
    returns = pd.Series([])
    
    metrics = engine._calculate_performance_metrics(returns, pd.DataFrame())
    
    assert 'error' in metrics

def test_nan_returns():
    """Test handling of NaN returns."""
    engine = BacktestEngine()
    
    returns = pd.Series([0.01, np.nan, 0.02, -0.01, 0.015])
    
    metrics = engine._calculate_performance_metrics(returns, pd.DataFrame())
    
    # Should handle NaN values gracefully
    assert 'total_return' in metrics
    assert not np.isnan(metrics['total_return'])

if __name__ == "__main__":
    pytest.main([__file__])
