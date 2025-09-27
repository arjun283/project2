"""
Feature engineering for commodity price prediction.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
from utils.constants import TECHNICAL_PARAMS

class FeatureEngineer:
    """Creates features for commodity price prediction models."""
    
    def __init__(self):
        self.feature_columns = []
    
    def create_features(
        self, 
        df: pd.DataFrame, 
        symbol: str,
        macro_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """Create comprehensive feature set for a commodity."""
        
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure we have required columns
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
        
        # Convert Date to datetime if needed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # 1. Price-based features
        df = self._create_price_features(df)
        
        # 2. Technical indicators
        df = self._create_technical_features(df)
        
        # 3. Statistical features
        df = self._create_statistical_features(df)
        
        # 4. Calendar features
        df = self._create_calendar_features(df)
        
        # 5. Cross-asset features
        if macro_data:
            df = self._create_cross_asset_features(df, symbol, macro_data)
        
        # 6. Carry features (futures spread proxy)
        df = self._create_carry_features(df, symbol)
        
        # 7. Regime features
        df = self._create_regime_features(df)
        
        # Store feature columns for later use
        self.feature_columns = [col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'data_quality']]
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features."""
        # Returns
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['returns_abs'] = np.abs(df['returns'])
        
        # Price levels
        df['price_high_20d'] = df['Close'].rolling(20).max()
        df['price_low_20d'] = df['Close'].rolling(20).min()
        df['price_position'] = (df['Close'] - df['price_low_20d']) / (df['price_high_20d'] - df['price_low_20d'])
        
        # Volatility
        df['vol_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['vol_5d'] = df['returns'].rolling(5).std() * np.sqrt(252)
        df['vol_ratio'] = df['vol_5d'] / df['vol_20d']
        
        return df
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features."""
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=TECHNICAL_PARAMS['rsi_period']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'], 
                            window_slow=TECHNICAL_PARAMS['macd_slow'],
                            window_fast=TECHNICAL_PARAMS['macd_fast'],
                            window_sign=TECHNICAL_PARAMS['macd_signal'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            df['High'], df['Low'], df['Close'], 
            window=TECHNICAL_PARAMS['atr_period']
        ).average_true_range()
        
        # Momentum
        df['momentum'] = ta.momentum.ROCIndicator(df['Close'], window=TECHNICAL_PARAMS['momentum_period']).roc()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['Close']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Moving averages
        df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=TECHNICAL_PARAMS['sma_period']).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['sma_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
        
        # Price vs moving averages
        df['price_vs_sma20'] = df['Close'] / df['sma_20'] - 1
        df['price_vs_sma50'] = df['Close'] / df['sma_50'] - 1
        df['price_vs_sma200'] = df['Close'] / df['sma_200'] - 1
        
        # Moving average slopes
        df['sma20_slope'] = df['sma_20'].diff(5) / 5
        df['sma50_slope'] = df['sma_50'].diff(10) / 10
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'mean_{window}d'] = df['returns'].rolling(window).mean()
            df[f'std_{window}d'] = df['returns'].rolling(window).std()
            df[f'skew_{window}d'] = df['returns'].rolling(window).skew()
            df[f'kurt_{window}d'] = df['returns'].rolling(window).kurt()
        
        # Z-scores
        df['zscore_20d'] = (df['returns'] - df['mean_20d']) / df['std_20d']
        
        # Autocorrelation
        df['autocorr_1d'] = df['returns'].rolling(20).apply(lambda x: x.autocorr(lag=1))
        df['autocorr_5d'] = df['returns'].rolling(20).apply(lambda x: x.autocorr(lag=5))
        
        # Hurst exponent (simplified)
        df['hurst'] = df['returns'].rolling(50).apply(self._calculate_hurst)
        
        return df
    
    def _create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar-based features."""
        if 'Date' not in df.columns:
            return df
        
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_month'] = df['Date'].dt.day
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['year'] = df['Date'].dt.year
        
        # Cyclical encoding
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Trading day of month
        df['trading_day_of_month'] = df.groupby(df['Date'].dt.to_period('M')).cumcount() + 1
        
        return df
    
    def _create_cross_asset_features(self, df: pd.DataFrame, symbol: str, macro_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create cross-asset features."""
        if 'Date' not in df.columns:
            return df
        
        # Merge macro data
        for macro_symbol, macro_df in macro_data.items():
            if macro_df.empty or 'Date' not in macro_df.columns:
                continue
            
            # Ensure Date column is datetime
            macro_df = macro_df.copy()
            macro_df['Date'] = pd.to_datetime(macro_df['Date'])
            
            # Merge on Date
            df = df.merge(macro_df[['Date', macro_symbol]], on='Date', how='left')
            
            # Create features from macro data
            if macro_symbol in df.columns:
                # Returns
                df[f'{macro_symbol}_returns'] = np.log(df[macro_symbol] / df[macro_symbol].shift(1))
                
                # Rolling correlation with commodity returns
                df[f'corr_{macro_symbol}_20d'] = df['returns'].rolling(20).corr(df[f'{macro_symbol}_returns'])
                
                # Rolling beta
                df[f'beta_{macro_symbol}_20d'] = df['returns'].rolling(20).cov(df[f'{macro_symbol}_returns']) / df[f'{macro_symbol}_returns'].rolling(20).var()
        
        return df
    
    def _create_carry_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create carry features (futures spread proxy)."""
        # For now, use a simple carry proxy based on price momentum
        # In practice, this would use front vs second month futures
        df['carry_proxy'] = df['Close'].pct_change(20)  # 20-day momentum as carry proxy
        
        # Term structure slope (simplified)
        df['term_slope'] = (df['Close'].shift(-5) / df['Close'] - 1) * 252 / 5  # Annualized 5-day forward return
        
        return df
    
    def _create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create regime detection features."""
        # Trend regime (based on SMA slope)
        df['trend_regime'] = np.where(df['sma20_slope'] > 0, 1, -1)
        
        # Volatility regime (based on vol percentiles)
        vol_20d = df['vol_20d'].rolling(100).rank(pct=True)
        df['vol_regime'] = pd.cut(vol_20d, bins=[0, 0.33, 0.67, 1], labels=['low', 'mid', 'high'])
        
        # Combined regime
        df['regime'] = df['trend_regime'].astype(str) + '_' + df['vol_regime'].astype(str)
        
        # Regime changes
        df['regime_change'] = (df['regime'] != df['regime'].shift(1)).astype(int)
        
        return df
    
    def _calculate_hurst(self, series: pd.Series) -> float:
        """Calculate Hurst exponent (simplified version)."""
        if len(series) < 10:
            return 0.5
        
        try:
            # Simplified Hurst calculation
            lags = range(2, min(20, len(series)//2))
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5
    
    def create_targets(self, df: pd.DataFrame, horizons: List[str]) -> pd.DataFrame:
        """Create target variables for different horizons."""
        df = df.copy()
        
        horizon_days = {'1d': 1, '5d': 5, '20d': 20}
        
        for horizon, days in horizon_days.items():
            # Future returns
            df[f'target_{horizon}'] = df['returns'].shift(-days)
            
            # Future price levels
            df[f'price_{horizon}'] = df['Close'].shift(-days)
            
            # Directional targets
            df[f'direction_{horizon}'] = np.where(df[f'target_{horizon}'] > 0, 1, 0)
        
        return df
    
    def get_feature_importance_data(self, df: pd.DataFrame, model=None) -> pd.DataFrame:
        """Get feature importance data for visualization."""
        if model is None or not hasattr(model, 'feature_importances_'):
            return pd.DataFrame()
        
        importance_data = []
        for i, feature in enumerate(self.feature_columns):
            if i < len(model.feature_importances_):
                importance_data.append({
                    'feature': feature,
                    'importance': model.feature_importances_[i]
                })
        
        return pd.DataFrame(importance_data).sort_values('importance', ascending=False)
