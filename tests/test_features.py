"""
Test feature engineering functionality.
"""

import pytest
import numpy as np
import pandas as pd
from services.features import FeatureEngineer

def test_feature_engineer_initialization():
    """Test FeatureEngineer initialization."""
    engineer = FeatureEngineer()
    assert engineer.feature_columns == []

def test_create_price_features():
    """Test price feature creation."""
    engineer = FeatureEngineer()
    
    # Create test data
    df = pd.DataFrame({
        'Close': [100, 101, 102, 101, 103, 104, 103, 105],
        'Date': pd.date_range('2023-01-01', periods=8)
    })
    
    result_df = engineer._create_price_features(df)
    
    # Check that returns are calculated
    assert 'returns' in result_df.columns
    assert 'returns_abs' in result_df.columns
    
    # Check that returns are log returns
    expected_returns = np.log(df['Close'] / df['Close'].shift(1))
    pd.testing.assert_series_equal(result_df['returns'], expected_returns, check_names=False)

def test_create_technical_features():
    """Test technical indicator creation."""
    engineer = FeatureEngineer()
    
    # Create test data
    df = pd.DataFrame({
        'Close': [100, 101, 102, 101, 103, 104, 103, 105, 106, 105, 107, 108, 107, 109, 110],
        'High': [101, 102, 103, 102, 104, 105, 104, 106, 107, 106, 108, 109, 108, 110, 111],
        'Low': [99, 100, 101, 100, 102, 103, 102, 104, 105, 104, 106, 107, 106, 108, 109],
        'Date': pd.date_range('2023-01-01', periods=15)
    })
    
    result_df = engineer._create_technical_features(df)
    
    # Check that technical indicators are created
    assert 'rsi' in result_df.columns
    assert 'macd' in result_df.columns
    assert 'atr' in result_df.columns
    assert 'sma_20' in result_df.columns

def test_create_calendar_features():
    """Test calendar feature creation."""
    engineer = FeatureEngineer()
    
    # Create test data
    df = pd.DataFrame({
        'Close': [100, 101, 102, 101, 103],
        'Date': pd.date_range('2023-01-01', periods=5)
    })
    
    result_df = engineer._create_calendar_features(df)
    
    # Check that calendar features are created
    assert 'day_of_week' in result_df.columns
    assert 'month' in result_df.columns
    assert 'quarter' in result_df.columns
    assert 'year' in result_df.columns

def test_create_statistical_features():
    """Test statistical feature creation."""
    engineer = FeatureEngineer()
    
    # Create test data
    df = pd.DataFrame({
        'Close': [100, 101, 102, 101, 103, 104, 103, 105, 106, 105, 107, 108, 107, 109, 110],
        'Date': pd.date_range('2023-01-01', periods=15)
    })
    
    # First create price features
    df = engineer._create_price_features(df)
    
    result_df = engineer._create_statistical_features(df)
    
    # Check that statistical features are created
    assert 'mean_5d' in result_df.columns
    assert 'std_5d' in result_df.columns
    assert 'skew_5d' in result_df.columns
    assert 'kurt_5d' in result_df.columns

def test_create_targets():
    """Test target creation."""
    engineer = FeatureEngineer()
    
    # Create test data
    df = pd.DataFrame({
        'Close': [100, 101, 102, 101, 103, 104, 103, 105, 106, 105, 107, 108, 107, 109, 110],
        'Date': pd.date_range('2023-01-01', periods=15)
    })
    
    # First create price features
    df = engineer._create_price_features(df)
    
    horizons = ['1d', '5d']
    result_df = engineer.create_targets(df, horizons)
    
    # Check that targets are created
    assert 'target_1d' in result_df.columns
    assert 'target_5d' in result_df.columns
    assert 'price_1d' in result_df.columns
    assert 'price_5d' in result_df.columns

def test_feature_statistics():
    """Test feature statistics calculation."""
    engineer = FeatureEngineer()
    
    # Create test data
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'feature3': [1, 1, 1, 1, 1]
    })
    
    stats = engineer.get_feature_statistics(df)
    
    # Check that statistics are calculated
    assert not stats.empty
    assert 'feature' in stats.columns
    assert 'mean' in stats.columns
    assert 'std' in stats.columns

def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    engineer = FeatureEngineer()
    
    df = pd.DataFrame()
    
    result_df = engineer.create_features(df, 'TEST')
    
    # Should return empty DataFrame
    assert result_df.empty

def test_missing_close_column():
    """Test handling of missing Close column."""
    engineer = FeatureEngineer()
    
    df = pd.DataFrame({
        'Open': [100, 101, 102],
        'Date': pd.date_range('2023-01-01', periods=3)
    })
    
    with pytest.raises(ValueError):
        engineer.create_features(df, 'TEST')

def test_feature_importance_data():
    """Test feature importance data extraction."""
    engineer = FeatureEngineer()
    
    # Mock model with feature_importances_
    class MockModel:
        def __init__(self):
            self.feature_importances_ = np.array([0.3, 0.2, 0.1])
            self.feature_names_in_ = ['feature1', 'feature2', 'feature3']
    
    model = MockModel()
    engineer.feature_columns = ['feature1', 'feature2', 'feature3']
    
    importance_df = engineer.get_feature_importance_data(pd.DataFrame(), model)
    
    # Check that importance data is extracted
    assert not importance_df.empty
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns

def test_hurst_calculation():
    """Test Hurst exponent calculation."""
    engineer = FeatureEngineer()
    
    # Create test series
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    hurst = engineer._calculate_hurst(series)
    
    # Hurst should be between 0 and 1
    assert 0 <= hurst <= 1

def test_short_series_hurst():
    """Test Hurst calculation with short series."""
    engineer = FeatureEngineer()
    
    # Create short series
    series = pd.Series([1, 2, 3])
    
    hurst = engineer._calculate_hurst(series)
    
    # Should return default value for short series
    assert hurst == 0.5

if __name__ == "__main__":
    pytest.main([__file__])
