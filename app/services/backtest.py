"""
Backtesting engine for commodity trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

from utils.constants import BACKTEST_PARAMS

class BacktestEngine:
    """Handles backtesting of trading strategies."""
    
    def __init__(self, transaction_cost_bps: float = 5.0):
        self.transaction_cost_bps = transaction_cost_bps / 10000  # Convert to decimal
        self.results = {}
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_config: Dict[str, Any],
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Run backtest for a single symbol."""
        
        if df.empty or 'Close' not in df.columns:
            return {'error': 'Invalid data'}
        
        # Filter by date range
        if start_date or end_date:
            df = self._filter_by_date(df, start_date, end_date)
        
        if df.empty:
            return {'error': 'No data in date range'}
        
        # Get strategy parameters
        horizon = strategy_config.get('horizon', '1d')
        threshold = strategy_config.get('threshold', 0.01)
        model_name = strategy_config.get('model', 'xgboost')
        
        # Get predictions
        predictions = self._get_predictions(df, symbol, horizon, model_name)
        
        if predictions is None:
            return {'error': 'Could not generate predictions'}
        
        # Generate signals
        signals = self._generate_signals(predictions, threshold)
        
        # Calculate returns
        returns = self._calculate_returns(df, signals)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(returns, df)
        
        # Store results
        self.results[symbol] = {
            'returns': returns,
            'signals': signals,
            'predictions': predictions,
            'metrics': metrics,
            'strategy_config': strategy_config
        }
        
        return self.results[symbol]
    
    def _filter_by_date(self, df: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if 'Date' not in df.columns:
            return df
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        if start_date:
            df = df[df['Date'] >= start_date]
        
        if end_date:
            df = df[df['Date'] <= end_date]
        
        return df
    
    def _get_predictions(self, df: pd.DataFrame, symbol: str, horizon: str, model_name: str) -> Optional[pd.Series]:
        """Get predictions for the strategy."""
        
        # For now, use a simple momentum-based prediction
        # In practice, this would use the trained models
        
        if f'target_{horizon}' in df.columns:
            # Use actual future returns as "predictions" for backtesting
            # In real implementation, these would be model predictions
            return df[f'target_{horizon}'].shift(1)  # Shift to avoid look-ahead bias
        
        # Fallback to simple momentum
        if horizon == '1d':
            return df['Close'].pct_change(1).shift(1)
        elif horizon == '5d':
            return df['Close'].pct_change(5).shift(5)
        elif horizon == '20d':
            return df['Close'].pct_change(20).shift(20)
        
        return None
    
    def _generate_signals(self, predictions: pd.Series, threshold: float) -> pd.Series:
        """Generate trading signals from predictions."""
        signals = pd.Series(0, index=predictions.index)
        
        # Long signal: prediction > threshold
        signals[predictions > threshold] = 1
        
        # Short signal: prediction < -threshold
        signals[predictions < -threshold] = -1
        
        return signals
    
    def _calculate_returns(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns with transaction costs."""
        
        # Align signals with price data
        if 'Date' in df.columns:
            df = df.set_index('Date')
        
        # Calculate price returns
        price_returns = df['Close'].pct_change()
        
        # Calculate strategy returns
        strategy_returns = signals.shift(1) * price_returns
        
        # Apply transaction costs
        signal_changes = (signals != signals.shift(1)).astype(int)
        transaction_costs = signal_changes * self.transaction_cost_bps
        
        # Net returns
        net_returns = strategy_returns - transaction_costs
        
        return net_returns.fillna(0)
    
    def _calculate_performance_metrics(self, returns: pd.Series, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        if returns.empty or returns.isnull().all():
            return {'error': 'No valid returns'}
        
        # Remove NaN values
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {'error': 'No valid returns after cleaning'}
        
        # Basic metrics
        total_return = (1 + returns_clean).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_clean)) - 1
        volatility = returns_clean.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = (1 + returns_clean).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Hit rate
        hit_rate = (returns_clean > 0).mean()
        
        # Additional metrics
        win_rate = (returns_clean > 0).mean()
        avg_win = returns_clean[returns_clean > 0].mean() if (returns_clean > 0).any() else 0
        avg_loss = returns_clean[returns_clean < 0].mean() if (returns_clean < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # VaR and CVaR
        var_95 = np.percentile(returns_clean, 5)
        cvar_95 = returns_clean[returns_clean <= var_95].mean()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'calmar_ratio': calmar_ratio,
            'n_trades': (returns_clean != 0).sum(),
            'n_winning_trades': (returns_clean > 0).sum(),
            'n_losing_trades': (returns_clean < 0).sum()
        }
    
    def run_portfolio_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio_config: Dict[str, Any],
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Run backtest for a portfolio of commodities."""
        
        if not data:
            return {'error': 'No data provided'}
        
        # Get portfolio weights
        weights = portfolio_config.get('weights', {})
        if not weights:
            # Equal weight by default
            symbols = list(data.keys())
            weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
        
        # Run individual backtests
        individual_results = {}
        for symbol, df in data.items():
            if symbol not in weights:
                continue
            
            strategy_config = portfolio_config.get('strategy', {})
            result = self.run_backtest(df, symbol, strategy_config, start_date, end_date)
            
            if 'error' not in result:
                individual_results[symbol] = result
        
        if not individual_results:
            return {'error': 'No valid individual results'}
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(individual_results, weights)
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_performance_metrics(portfolio_returns, pd.DataFrame())
        
        # Add portfolio-specific metrics
        portfolio_metrics['n_assets'] = len(individual_results)
        portfolio_metrics['weights'] = weights
        
        return {
            'portfolio_returns': portfolio_returns,
            'portfolio_metrics': portfolio_metrics,
            'individual_results': individual_results
        }
    
    def _calculate_portfolio_returns(
        self, 
        individual_results: Dict[str, Any], 
        weights: Dict[str, float]
    ) -> pd.Series:
        """Calculate portfolio returns from individual results."""
        
        # Get all return series
        return_series = {}
        for symbol, result in individual_results.items():
            if 'returns' in result:
                return_series[symbol] = result['returns']
        
        if not return_series:
            return pd.Series(dtype=float)
        
        # Align all series to common index
        aligned_returns = pd.DataFrame(return_series).fillna(0)
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0, index=aligned_returns.index)
        
        for symbol, returns in aligned_returns.items():
            if symbol in weights:
                portfolio_returns += weights[symbol] * returns
        
        return portfolio_returns
    
    def get_equity_curve(self, returns: pd.Series, initial_capital: float = 10000) -> pd.Series:
        """Calculate equity curve from returns."""
        if returns.empty:
            return pd.Series(dtype=float)
        
        cumulative_returns = (1 + returns).cumprod()
        equity_curve = initial_capital * cumulative_returns
        
        return equity_curve
    
    def get_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        if returns.empty:
            return pd.Series(dtype=float)
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown
    
    def get_rolling_metrics(
        self, 
        returns: pd.Series, 
        window: int = 252
    ) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        
        if returns.empty:
            return pd.DataFrame()
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        rolling_returns = returns.rolling(window)
        rolling_metrics['rolling_return'] = rolling_returns.mean() * 252
        rolling_metrics['rolling_volatility'] = rolling_returns.std() * np.sqrt(252)
        rolling_metrics['rolling_sharpe'] = rolling_metrics['rolling_return'] / rolling_metrics['rolling_volatility']
        
        # Rolling drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.rolling(window).max()
        rolling_metrics['rolling_drawdown'] = (cumulative_returns - rolling_max) / rolling_max
        
        return rolling_metrics.dropna()
    
    def compare_strategies(
        self, 
        strategy_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Compare multiple strategies."""
        
        comparison_data = []
        
        for strategy_name, results in strategy_results.items():
            if 'metrics' in results and 'error' not in results['metrics']:
                metrics = results['metrics']
                comparison_data.append({
                    'strategy': strategy_name,
                    'annualized_return': metrics.get('annualized_return', 0),
                    'volatility': metrics.get('volatility', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'hit_rate': metrics.get('hit_rate', 0),
                    'calmar_ratio': metrics.get('calmar_ratio', 0)
                })
        
        return pd.DataFrame(comparison_data).sort_values('sharpe_ratio', ascending=False)
