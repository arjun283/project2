"""
Date and time utilities.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional

def get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Get trading days between start and end dates (excludes weekends)."""
    return pd.bdate_range(start=start_date, end=end_date)

def get_lookback_date(days: int, end_date: Optional[str] = None) -> str:
    """Get date N days back from end_date (or today)."""
    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end_date)
    
    lookback_date = end_date - timedelta(days=days)
    return lookback_date.strftime('%Y-%m-%d')

def align_to_trading_days(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """Align DataFrame to trading days only."""
    if date_col not in df.columns:
        return df
    
    # Convert to datetime if needed
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create trading day range
    start_date = df[date_col].min()
    end_date = df[date_col].max()
    trading_days = get_trading_days(start_date, end_date)
    
    # Set date as index and reindex to trading days
    df_indexed = df.set_index(date_col)
    df_aligned = df_indexed.reindex(trading_days)
    
    # Reset index to get date column back
    df_aligned = df_aligned.reset_index()
    df_aligned.rename(columns={'index': date_col}, inplace=True)
    
    return df_aligned

def get_quarter_dates(year: int) -> Tuple[str, str, str, str]:
    """Get start dates for each quarter of a year."""
    return (
        f"{year}-01-01",  # Q1
        f"{year}-04-01",  # Q2
        f"{year}-07-01",  # Q3
        f"{year}-10-01",  # Q4
    )

def is_weekend(date: str) -> bool:
    """Check if date is a weekend."""
    dt = pd.to_datetime(date)
    return dt.weekday() >= 5

def get_next_trading_day(date: str) -> str:
    """Get next trading day after given date."""
    dt = pd.to_datetime(date)
    next_day = dt + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    
    return next_day.strftime('%Y-%m-%d')
