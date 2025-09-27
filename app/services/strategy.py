"""
Trading strategy implementation and simulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

from utils.constants import HORIZONS, BACKTEST_PARAMS

class StrategySimulator:
    """Simulates trading strategies for commodities."""
    
    def __init__(self, transaction_cost_bps: float = 5.0):
        self.transaction_cost_bps = transaction_cost_bps / 10000
        self.strategies = {}
        self.portfolio = {}
    
    def create_strategy(
        self,
        name: str,
        symbol: str,
        horizon: str,
        threshold: float,
        model_name: str = 'xgboost',
        max_position: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create a trading strategy configuration."""
        
        strategy = {
            'name': name,
            'symbol': symbol,
            'horizon': horizon,
            'threshold': threshold,
            'model_name': model_name,
            'max_position': max_position,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'created_at': datetime.now().isoformat()
        }
        
        self.strategies[name] = strategy
        return strategy
    
    def simulate_strategy(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        initial_capital: float = 10000
    ) -> Dict[str, Any]:
        """Simulate a trading strategy."""
        
        if strategy_name not in self.strategies:
            return {'error': f'Strategy {strategy_name} not found'}
        
        strategy = self.strategies[strategy_name]
        
        if df.empty or 'Close' not in df.columns:
            return {'error': 'Invalid data'}
        
        # Generate signals
        signals = self._generate_signals(df, strategy)
        
        # Calculate positions
        positions = self._calculate_positions(signals, strategy)
        
        # Calculate returns
        returns = self._calculate_strategy_returns(df, positions, strategy)
        
        # Calculate equity curve
        equity_curve = self._calculate_equity_curve(returns, initial_capital)
        
        # Calculate performance metrics
        metrics = self._calculate_strategy_metrics(returns, equity_curve)
        
        # Calculate trade statistics
        trade_stats = self._calculate_trade_statistics(positions, returns)
        
        return {
            'strategy': strategy,
            'signals': signals,
            'positions': positions,
            'returns': returns,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'trade_stats': trade_stats,
            'initial_capital': initial_capital
        }
    
    def _generate_signals(self, df: pd.DataFrame, strategy: Dict[str, Any]) -> pd.Series:
        """Generate trading signals based on strategy configuration."""
        
        horizon = strategy['horizon']
        threshold = strategy['threshold']
        
        # Get predictions (simplified - in practice would use trained models)
        if f'target_{horizon}' in df.columns:
            predictions = df[f'target_{horizon}'].shift(1)  # Avoid look-ahead bias
        else:
            # Fallback to simple momentum
            horizon_days = HORIZONS[horizon]['days']
            predictions = df['Close'].pct_change(horizon_days).shift(horizon_days)
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Long signal: prediction > threshold
        signals[predictions > threshold] = 1
        
        # Short signal: prediction < -threshold
        signals[predictions < -threshold] = -1
        
        return signals
    
    def _calculate_positions(self, signals: pd.Series, strategy: Dict[str, Any]) -> pd.Series:
        """Calculate position sizes based on signals and risk management."""
        
        max_position = strategy.get('max_position', 1.0)
        positions = signals * max_position
        
        # Apply stop loss and take profit (simplified)
        stop_loss = strategy.get('stop_loss')
        take_profit = strategy.get('take_profit')
        
        if stop_loss or take_profit:
            # This is a simplified implementation
            # In practice, would track entry prices and apply stops
            pass
        
        return positions
    
    def _calculate_strategy_returns(
        self, 
        df: pd.DataFrame, 
        positions: pd.Series, 
        strategy: Dict[str, Any]
    ) -> pd.Series:
        """Calculate strategy returns with transaction costs."""
        
        # Calculate price returns
        price_returns = df['Close'].pct_change()
        
        # Calculate strategy returns
        strategy_returns = positions.shift(1) * price_returns
        
        # Apply transaction costs
        position_changes = (positions != positions.shift(1)).astype(int)
        transaction_costs = position_changes * self.transaction_cost_bps
        
        # Net returns
        net_returns = strategy_returns - transaction_costs
        
        return net_returns.fillna(0)
    
    def _calculate_equity_curve(self, returns: pd.Series, initial_capital: float) -> pd.Series:
        """Calculate equity curve from returns."""
        
        if returns.empty:
            return pd.Series(dtype=float)
        
        cumulative_returns = (1 + returns).cumprod()
        equity_curve = initial_capital * cumulative_returns
        
        return equity_curve
    
    def _calculate_strategy_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive strategy metrics."""
        
        if returns.empty or returns.isnull().all():
            return {'error': 'No valid returns'}
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {'error': 'No valid returns after cleaning'}
        
        # Basic performance metrics
        total_return = (1 + returns_clean).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_clean)) - 1
        volatility = returns_clean.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = (1 + returns_clean).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        hit_rate = (returns_clean > 0).mean()
        win_rate = (returns_clean > 0).mean()
        avg_win = returns_clean[returns_clean > 0].mean() if (returns_clean > 0).any() else 0
        avg_loss = returns_clean[returns_clean < 0].mean() if (returns_clean < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Risk metrics
        var_95 = np.percentile(returns_clean, 5)
        cvar_95 = returns_clean[returns_clean <= var_95].mean()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_clean[returns_clean < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
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
    
    def _calculate_trade_statistics(self, positions: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Calculate detailed trade statistics."""
        
        if positions.empty or returns.empty:
            return {'error': 'No position or return data'}
        
        # Find trade entries and exits
        position_changes = (positions != positions.shift(1)).astype(int)
        trade_entries = position_changes[position_changes == 1]
        trade_exits = position_changes[position_changes == 1].shift(1)
        
        # Calculate trade returns
        trade_returns = []
        trade_durations = []
        
        for entry_idx in trade_entries.index:
            if entry_idx in returns.index:
                # Find next exit
                next_exits = trade_exits[trade_exits.index > entry_idx]
                if not next_exits.empty:
                    exit_idx = next_exits.index[0]
                    trade_return = returns.loc[entry_idx:exit_idx].sum()
                    trade_duration = (exit_idx - entry_idx).days if hasattr(exit_idx, 'days') else 1
                    
                    trade_returns.append(trade_return)
                    trade_durations.append(trade_duration)
        
        if not trade_returns:
            return {'error': 'No complete trades found'}
        
        trade_returns = np.array(trade_returns)
        trade_durations = np.array(trade_durations)
        
        return {
            'n_trades': len(trade_returns),
            'avg_trade_return': np.mean(trade_returns),
            'median_trade_return': np.median(trade_returns),
            'std_trade_return': np.std(trade_returns),
            'best_trade': np.max(trade_returns),
            'worst_trade': np.min(trade_returns),
            'avg_trade_duration': np.mean(trade_durations),
            'winning_trades': np.sum(trade_returns > 0),
            'losing_trades': np.sum(trade_returns < 0),
            'win_rate': np.mean(trade_returns > 0),
            'avg_winning_trade': np.mean(trade_returns[trade_returns > 0]) if np.any(trade_returns > 0) else 0,
            'avg_losing_trade': np.mean(trade_returns[trade_returns < 0]) if np.any(trade_returns < 0) else 0
        }
    
    def create_portfolio(
        self,
        name: str,
        strategies: List[str],
        weights: Optional[Dict[str, float]] = None,
        rebalance_frequency: str = 'monthly'
    ) -> Dict[str, Any]:
        """Create a portfolio of strategies."""
        
        if not strategies:
            return {'error': 'No strategies provided'}
        
        # Default to equal weights
        if weights is None:
            weights = {strategy: 1.0 / len(strategies) for strategy in strategies}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {strategy: weight / total_weight for strategy, weight in weights.items()}
        
        portfolio = {
            'name': name,
            'strategies': strategies,
            'weights': weights,
            'rebalance_frequency': rebalance_frequency,
            'created_at': datetime.now().isoformat()
        }
        
        self.portfolio[name] = portfolio
        return portfolio
    
    def simulate_portfolio(
        self,
        portfolio_name: str,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = 10000
    ) -> Dict[str, Any]:
        """Simulate a portfolio of strategies."""
        
        if portfolio_name not in self.portfolio:
            return {'error': f'Portfolio {portfolio_name} not found'}
        
        portfolio = self.portfolio[portfolio_name]
        strategies = portfolio['strategies']
        weights = portfolio['weights']
        
        # Simulate individual strategies
        strategy_results = {}
        for strategy_name in strategies:
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                symbol = strategy['symbol']
                
                if symbol in data:
                    result = self.simulate_strategy(strategy_name, data[symbol], initial_capital)
                    if 'error' not in result:
                        strategy_results[strategy_name] = result
        
        if not strategy_results:
            return {'error': 'No valid strategy results'}
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(strategy_results, weights)
        
        # Calculate portfolio equity curve
        portfolio_equity = self._calculate_equity_curve(portfolio_returns, initial_capital)
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_strategy_metrics(portfolio_returns, portfolio_equity)
        
        return {
            'portfolio': portfolio,
            'strategy_results': strategy_results,
            'portfolio_returns': portfolio_returns,
            'portfolio_equity': portfolio_equity,
            'portfolio_metrics': portfolio_metrics,
            'initial_capital': initial_capital
        }
    
    def _calculate_portfolio_returns(
        self, 
        strategy_results: Dict[str, Any], 
        weights: Dict[str, float]
    ) -> pd.Series:
        """Calculate portfolio returns from individual strategy results."""
        
        # Get all return series
        return_series = {}
        for strategy_name, result in strategy_results.items():
            if 'returns' in result:
                return_series[strategy_name] = result['returns']
        
        if not return_series:
            return pd.Series(dtype=float)
        
        # Align all series to common index
        aligned_returns = pd.DataFrame(return_series).fillna(0)
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0, index=aligned_returns.index)
        
        for strategy_name, returns in aligned_returns.items():
            if strategy_name in weights:
                portfolio_returns += weights[strategy_name] * returns
        
        return portfolio_returns
    
    def get_strategy_summary(self) -> pd.DataFrame:
        """Get summary of all strategies."""
        
        summary_data = []
        for name, strategy in self.strategies.items():
            summary_data.append({
                'name': name,
                'symbol': strategy['symbol'],
                'horizon': strategy['horizon'],
                'threshold': strategy['threshold'],
                'model': strategy['model_name'],
                'max_position': strategy['max_position'],
                'created_at': strategy['created_at']
            })
        
        return pd.DataFrame(summary_data)
    
    def get_portfolio_summary(self) -> pd.DataFrame:
        """Get summary of all portfolios."""
        
        summary_data = []
        for name, portfolio in self.portfolio.items():
            summary_data.append({
                'name': name,
                'strategies': ', '.join(portfolio['strategies']),
                'n_strategies': len(portfolio['strategies']),
                'rebalance_frequency': portfolio['rebalance_frequency'],
                'created_at': portfolio['created_at']
            })
        
        return pd.DataFrame(summary_data)
