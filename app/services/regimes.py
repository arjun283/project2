"""
Regime detection and analysis for commodity markets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.constants import REGIMES

class RegimeDetector:
    """Detects and analyzes market regimes for commodities."""
    
    def __init__(self, lookback_days: int = 100):
        self.lookback_days = lookback_days
        self.regime_history = {}
    
    def detect_regimes(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Detect regimes for a commodity time series."""
        if df.empty or 'Close' not in df.columns:
            return df
        
        df = df.copy()
        
        # Calculate trend signal (SMA slope)
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_slope'] = df['sma_20'].diff(5) / 5  # 5-day slope
        df['trend_signal'] = np.where(df['sma_slope'] > 0, 1, -1)
        
        # Calculate volatility signal
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['vol_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Volatility percentiles
        df['vol_percentile'] = df['vol_20d'].rolling(self.lookback_days).rank(pct=True)
        df['vol_signal'] = pd.cut(
            df['vol_percentile'], 
            bins=[0, 0.33, 0.67, 1], 
            labels=['low', 'mid', 'high']
        )
        
        # Combine signals into regimes
        df['regime'] = self._combine_signals(df['trend_signal'], df['vol_signal'])
        
        # Detect regime changes
        df['regime_change'] = (df['regime'] != df['regime'].shift(1)).astype(int)
        
        # Store regime history
        self.regime_history[symbol] = df[['Date', 'regime', 'regime_change']].copy()
        
        return df
    
    def _combine_signals(self, trend_signal: pd.Series, vol_signal: pd.Series) -> pd.Series:
        """Combine trend and volatility signals into regimes."""
        regimes = []
        
        for trend, vol in zip(trend_signal, vol_signal):
            if pd.isna(trend) or pd.isna(vol):
                regimes.append('unknown')
            elif trend == 1 and vol == 'low':
                regimes.append('bull_quiet')
            elif trend == 1 and vol == 'high':
                regimes.append('bull_volatile')
            elif trend == -1 and vol == 'low':
                regimes.append('bear_quiet')
            elif trend == -1 and vol == 'high':
                regimes.append('bear_volatile')
            elif trend == 1 and vol == 'mid':
                regimes.append('sideways_quiet')
            elif trend == -1 and vol == 'mid':
                regimes.append('sideways_volatile')
            else:
                regimes.append('sideways_quiet')
        
        return pd.Series(regimes, index=trend_signal.index)
    
    def get_current_regime(self, symbol: str) -> Optional[str]:
        """Get current regime for a symbol."""
        if symbol not in self.regime_history:
            return None
        
        history = self.regime_history[symbol]
        if history.empty:
            return None
        
        return history['regime'].iloc[-1]
    
    def get_regime_transitions(self, symbol: str, days_back: int = 30) -> List[Dict]:
        """Get recent regime transitions for a symbol."""
        if symbol not in self.regime_history:
            return []
        
        history = self.regime_history[symbol]
        if history.empty:
            return []
        
        # Get recent data
        recent = history.tail(days_back)
        
        # Find regime changes
        changes = recent[recent['regime_change'] == 1]
        
        transitions = []
        for idx, row in changes.iterrows():
            transitions.append({
                'date': row['Date'],
                'regime': row['regime'],
                'regime_label': REGIMES.get(row['regime'], row['regime'])
            })
        
        return transitions
    
    def get_regime_stats(self, symbol: str) -> Dict:
        """Get regime statistics for a symbol."""
        if symbol not in self.regime_history:
            return {}
        
        history = self.regime_history[symbol]
        if history.empty:
            return {}
        
        # Count regime occurrences
        regime_counts = history['regime'].value_counts()
        
        # Calculate regime durations
        regime_durations = []
        current_regime = None
        start_date = None
        
        for _, row in history.iterrows():
            if row['regime'] != current_regime:
                if current_regime is not None and start_date is not None:
                    duration = (row['Date'] - start_date).days
                    regime_durations.append({
                        'regime': current_regime,
                        'duration_days': duration,
                        'start_date': start_date,
                        'end_date': row['Date']
                    })
                
                current_regime = row['regime']
                start_date = row['Date']
        
        # Add last regime if it's still ongoing
        if current_regime is not None and start_date is not None:
            duration = (history['Date'].iloc[-1] - start_date).days
            regime_durations.append({
                'regime': current_regime,
                'duration_days': duration,
                'start_date': start_date,
                'end_date': history['Date'].iloc[-1]
            })
        
        return {
            'regime_counts': regime_counts.to_dict(),
            'regime_durations': regime_durations,
            'total_days': len(history),
            'regime_changes': history['regime_change'].sum()
        }
    
    def get_regime_performance(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Get performance metrics by regime."""
        if df.empty or 'regime' not in df.columns:
            return pd.DataFrame()
        
        # Calculate returns
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Group by regime and calculate metrics
        regime_stats = []
        
        for regime in df['regime'].unique():
            if pd.isna(regime) or regime == 'unknown':
                continue
            
            regime_data = df[df['regime'] == regime]
            if len(regime_data) < 5:  # Need minimum data points
                continue
            
            returns = regime_data['returns'].dropna()
            if len(returns) == 0:
                continue
            
            # Calculate performance metrics
            total_return = returns.sum()
            annualized_return = total_return * (252 / len(returns))
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(returns)
            hit_rate = (returns > 0).mean()
            
            regime_stats.append({
                'regime': regime,
                'regime_label': REGIMES.get(regime, regime),
                'days': len(regime_data),
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'hit_rate': hit_rate,
                'avg_daily_return': returns.mean(),
                'positive_days': (returns > 0).sum(),
                'negative_days': (returns < 0).sum()
            })
        
        return pd.DataFrame(regime_stats).sort_values('sharpe_ratio', ascending=False)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from a returns series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def get_regime_correlation_matrix(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Get correlation matrix of returns by regime across symbols."""
        regime_returns = {}
        
        for symbol, df in data.items():
            if 'regime' not in df.columns or 'returns' not in df.columns:
                continue
            
            for regime in df['regime'].unique():
                if pd.isna(regime) or regime == 'unknown':
                    continue
                
                regime_data = df[df['regime'] == regime]
                if len(regime_data) < 10:  # Need minimum data points
                    continue
                
                key = f"{symbol}_{regime}"
                regime_returns[key] = regime_data['returns'].dropna()
        
        if not regime_returns:
            return pd.DataFrame()
        
        # Create DataFrame and calculate correlation
        returns_df = pd.DataFrame(regime_returns)
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def plot_regime_timeline(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Prepare data for regime timeline visualization."""
        if 'regime' not in df.columns or 'Date' not in df.columns:
            return {}
        
        # Create regime timeline data
        timeline_data = []
        current_regime = None
        start_date = None
        
        for _, row in df.iterrows():
            if row['regime'] != current_regime:
                if current_regime is not None and start_date is not None:
                    timeline_data.append({
                        'start': start_date,
                        'end': row['Date'],
                        'regime': current_regime,
                        'regime_label': REGIMES.get(current_regime, current_regime)
                    })
                
                current_regime = row['regime']
                start_date = row['Date']
        
        # Add last regime
        if current_regime is not None and start_date is not None:
            timeline_data.append({
                'start': start_date,
                'end': df['Date'].iloc[-1],
                'regime': current_regime,
                'regime_label': REGIMES.get(current_regime, current_regime)
            })
        
        return {
            'timeline': timeline_data,
            'regime_colors': {
                'bull_quiet': '#2E8B57',
                'bull_volatile': '#32CD32',
                'bear_quiet': '#DC143C',
                'bear_volatile': '#FF6347',
                'sideways_quiet': '#D3D3D3',
                'sideways_volatile': '#A9A9A9'
            }
        }
