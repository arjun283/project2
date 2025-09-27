"""
Constants and configuration for the Commodity Price Predictor app.
"""

# Commodity symbols and metadata
COMMODITIES = {
    'CL=F': {'name': 'WTI Crude Oil', 'category': 'Energy', 'currency': 'USD'},
    'BZ=F': {'name': 'Brent Crude Oil', 'category': 'Energy', 'currency': 'USD'},
    'NG=F': {'name': 'Natural Gas', 'category': 'Energy', 'currency': 'USD'},
    'GC=F': {'name': 'Gold', 'category': 'Metals', 'currency': 'USD'},
    'SI=F': {'name': 'Silver', 'category': 'Metals', 'currency': 'USD'},
    'HG=F': {'name': 'Copper', 'category': 'Metals', 'currency': 'USD'},
    'ZC=F': {'name': 'Corn', 'category': 'Agriculture', 'currency': 'USD'},
    'ZS=F': {'name': 'Soybeans', 'category': 'Agriculture', 'currency': 'USD'},
}

# Macro indicators
MACRO_INDICATORS = {
    'UUP': {'name': 'Dollar Index Proxy', 'source': 'yfinance'},
    'SPY': {'name': 'S&P 500', 'source': 'yfinance'},
    'DGS10': {'name': '10-Year Treasury Yield', 'source': 'fred'},
}

# Prediction horizons
HORIZONS = {
    '1d': {'days': 1, 'label': '1 Day'},
    '5d': {'days': 5, 'label': '5 Days'},
    '20d': {'days': 20, 'label': '20 Days'},
}

# Model types
MODEL_TYPES = {
    'naive_last': 'Last Value',
    'naive_seasonal': 'Seasonal Naive',
    'arima': 'ARIMA/SARIMAX',
    'xgboost': 'XGBoost',
    'prophet': 'Prophet (Optional)',
}

# Regime types
REGIMES = {
    'bull_quiet': 'Bull Quiet',
    'bull_volatile': 'Bull Volatile', 
    'bear_quiet': 'Bear Quiet',
    'bear_volatile': 'Bear Volatile',
    'sideways_quiet': 'Sideways Quiet',
    'sideways_volatile': 'Sideways Volatile',
}

# Technical indicators parameters
TECHNICAL_PARAMS = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'atr_period': 14,
    'momentum_period': 10,
    'sma_period': 20,
    'vol_period': 20,
}

# Backtesting parameters
BACKTEST_PARAMS = {
    'min_training_days': 252 * 2,  # 2 years minimum
    'step_days': 5,  # 5-day steps
    'max_order_arima': (3, 2, 3),  # Max ARIMA order
    'transaction_cost_bps': 5,  # 5 bps default
}

# Data quality thresholds
DATA_QUALITY = {
    'max_gap_days': 5,  # Max gap to forward fill
    'min_data_points': 100,  # Min points for modeling
    'max_missing_pct': 0.1,  # Max 10% missing data
}

# API configuration
API_CONFIG = {
    'host': '127.0.0.1',
    'port': 8000,
    'title': 'Commodity Price Predictor API',
    'version': '1.0.0',
}
