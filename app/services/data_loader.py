"""
Data loading service for commodity prices and macro indicators.
"""

import os
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.constants import COMMODITIES, MACRO_INDICATORS
from utils.guardrails import validate_data_quality, clean_data, detect_data_gaps
from utils.dates import get_lookback_date, align_to_trading_days

class DataLoader:
    """Handles data loading from various sources with caching and fallbacks."""
    
    def __init__(self, data_dir: str = "data", use_cache: bool = True):
        self.data_dir = data_dir
        self.use_cache = use_cache
        self.fred_api_key = os.getenv('FRED_API_KEY')
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_commodity_data(
        self, 
        symbols: List[str], 
        start_date: str = None, 
        end_date: str = None,
        use_cached: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Load commodity price data with caching and fallback to CSV."""
        
        if start_date is None:
            start_date = get_lookback_date(365 * 5)  # 5 years back
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        data = {}
        
        for symbol in symbols:
            try:
                # Try to load from yfinance first
                if not use_cached:
                    df = self._load_from_yfinance(symbol, start_date, end_date)
                    if df is not None and not df.empty:
                        data[symbol] = df
                        # Save to cache
                        self._save_to_cache(df, symbol)
                        continue
                
                # Try to load from cache
                if use_cached:
                    df = self._load_from_cache(symbol)
                    if df is not None and not df.empty:
                        data[symbol] = df
                        continue
                
                # If both fail, show warning
                st.warning(f"Could not load data for {symbol}")
                
            except Exception as e:
                st.warning(f"Error loading {symbol}: {str(e)}")
                continue
        
        return data
    
    def _load_from_yfinance(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            # Reset index to get Date column
            df = df.reset_index()
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
            # Align to trading days
            df = align_to_trading_days(df)
            
            # Clean and validate data
            df = clean_data(df, symbol)
            df = detect_data_gaps(df)
            
            # Validate quality
            quality = validate_data_quality(df, symbol)
            if not quality['is_valid']:
                st.warning(f"Data quality issues for {symbol}: {quality['issues']}")
            
            return df
            
        except Exception as e:
            print(f"Error loading {symbol} from yfinance: {e}")
            return None
    
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from cached CSV file."""
        cache_file = os.path.join(self.data_dir, f"{symbol}.csv")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            df = pd.read_csv(cache_file)
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            return df
        except Exception as e:
            print(f"Error loading {symbol} from cache: {e}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str) -> None:
        """Save data to cache."""
        if not self.use_cache:
            return
        
        cache_file = os.path.join(self.data_dir, f"{symbol}.csv")
        try:
            df.to_csv(cache_file, index=False)
        except Exception as e:
            print(f"Error saving {symbol} to cache: {e}")
    
    @st.cache_data(ttl=3600)
    def load_macro_data(
        self, 
        indicators: List[str] = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """Load macroeconomic indicators."""
        
        if indicators is None:
            indicators = list(MACRO_INDICATORS.keys())
        
        if start_date is None:
            start_date = get_lookback_date(365 * 5)
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        data = {}
        
        for indicator in indicators:
            try:
                if indicator in ['UUP', 'SPY']:
                    # Load from yfinance
                    df = self._load_macro_from_yfinance(indicator, start_date, end_date)
                elif indicator == 'DGS10' and self.fred_api_key:
                    # Load from FRED
                    df = self._load_macro_from_fred(indicator, start_date, end_date)
                else:
                    st.warning(f"Macro indicator {indicator} not available")
                    continue
                
                if df is not None and not df.empty:
                    data[indicator] = df
                
            except Exception as e:
                st.warning(f"Error loading macro indicator {indicator}: {str(e)}")
                continue
        
        return data
    
    def _load_macro_from_yfinance(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load macro data from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            # Use Close price as the indicator value
            df = df[['Close']].reset_index()
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df.rename(columns={'Close': symbol}, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error loading {symbol} from yfinance: {e}")
            return None
    
    def _load_macro_from_fred(self, series_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load macro data from FRED API."""
        if not self.fred_api_key:
            return None
        
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date,
                'frequency': 'd'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            observations = data.get('observations', [])
            
            if not observations:
                return None
            
            # Convert to DataFrame
            df_data = []
            for obs in observations:
                if obs['value'] != '.':
                    df_data.append({
                        'Date': obs['date'],
                        series_id: float(obs['value'])
                    })
            
            df = pd.DataFrame(df_data)
            return df
            
        except Exception as e:
            print(f"Error loading {series_id} from FRED: {e}")
            return None
    
    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Get summary statistics for loaded data."""
        summary_data = []
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            summary_data.append({
                'Symbol': symbol,
                'Name': COMMODITIES.get(symbol, {}).get('name', symbol),
                'Category': COMMODITIES.get(symbol, {}).get('category', 'Unknown'),
                'Start Date': df['Date'].min(),
                'End Date': df['Date'].max(),
                'Data Points': len(df),
                'Missing %': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100).round(1),
                'Last Close': df['Close'].iloc[-1] if 'Close' in df.columns else None,
                'Quality': 'Good' if validate_data_quality(df, symbol)['is_valid'] else 'Issues'
            })
        
        return pd.DataFrame(summary_data)
    
    def refresh_data(self, symbols: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Force refresh data from sources (bypass cache)."""
        return self.load_commodity_data(symbols, start_date, end_date, use_cached=False)
