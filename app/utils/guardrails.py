"""
Data quality guardrails and validation utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .constants import DATA_QUALITY

def validate_data_quality(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Validate data quality and return quality metrics."""
    if df.empty:
        return {
            'is_valid': False,
            'issues': ['Empty dataset'],
            'quality_score': 0.0
        }
    
    issues = []
    quality_score = 1.0
    
    # Check minimum data points
    if len(df) < DATA_QUALITY['min_data_points']:
        issues.append(f"Insufficient data points: {len(df)} < {DATA_QUALITY['min_data_points']}")
        quality_score -= 0.3
    
    # Check for missing values
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_pct > DATA_QUALITY['max_missing_pct']:
        issues.append(f"Too many missing values: {missing_pct:.1%} > {DATA_QUALITY['max_missing_pct']:.1%}")
        quality_score -= 0.2
    
    # Check for price columns
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_price_cols = [col for col in price_cols if col not in df.columns]
    if missing_price_cols:
        issues.append(f"Missing price columns: {missing_price_cols}")
        quality_score -= 0.3
    
    # Check for negative prices
    if 'Close' in df.columns:
        negative_prices = (df['Close'] <= 0).sum()
        if negative_prices > 0:
            issues.append(f"Negative or zero prices: {negative_prices} instances")
            quality_score -= 0.2
    
    # Check for extreme outliers
    if 'Close' in df.columns:
        returns = df['Close'].pct_change().dropna()
        if len(returns) > 0:
            extreme_returns = (abs(returns) > 0.5).sum()  # >50% daily moves
            if extreme_returns > len(returns) * 0.01:  # >1% of days
                issues.append(f"Extreme price moves: {extreme_returns} instances")
                quality_score -= 0.1
    
    return {
        'is_valid': quality_score > 0.5,
        'issues': issues,
        'quality_score': max(0.0, quality_score),
        'missing_pct': missing_pct,
        'data_points': len(df)
    }

def detect_data_gaps(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """Detect and mark data gaps in time series."""
    if date_col not in df.columns:
        return df
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Calculate expected vs actual gaps
    df['days_since_last'] = df[date_col].diff().dt.days
    df['data_quality'] = 'good'
    
    # Mark gaps
    large_gaps = df['days_since_last'] > DATA_QUALITY['max_gap_days']
    df.loc[large_gaps, 'data_quality'] = 'gap'
    
    # Mark missing data
    missing_data = df.isnull().any(axis=1)
    df.loc[missing_data, 'data_quality'] = 'missing'
    
    return df

def clean_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Clean and prepare data for analysis."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Forward fill small gaps
    df = df.fillna(method='ffill', limit=DATA_QUALITY['max_gap_days'])
    
    # Remove rows with all NaN
    df = df.dropna(how='all')
    
    # Ensure numeric columns are numeric
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove extreme outliers (beyond 5 standard deviations)
    if 'Close' in df.columns:
        returns = df['Close'].pct_change()
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        extreme_outliers = z_scores > 5
        if extreme_outliers.any():
            df = df[~extreme_outliers]
    
    return df

def validate_model_inputs(X: pd.DataFrame, y: pd.Series) -> Tuple[bool, str]:
    """Validate inputs for model training."""
    if X.empty or y.empty:
        return False, "Empty input data"
    
    if len(X) != len(y):
        return False, f"Feature and target length mismatch: {len(X)} vs {len(y)}"
    
    if X.isnull().any().any():
        return False, "Features contain missing values"
    
    if y.isnull().any():
        return False, "Target contains missing values"
    
    if len(X) < 50:
        return False, f"Insufficient data for training: {len(X)} < 50"
    
    return True, "Valid inputs"

def check_data_freshness(df: pd.DataFrame, max_days_old: int = 7) -> bool:
    """Check if data is fresh enough for current analysis."""
    if df.empty or 'Date' not in df.columns:
        return False
    
    latest_date = pd.to_datetime(df['Date']).max()
    days_old = (pd.Timestamp.now() - latest_date).days
    
    return days_old <= max_days_old
