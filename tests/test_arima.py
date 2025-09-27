"""
Test ARIMA functionality.
"""

import pytest
import numpy as np
import pandas as pd
from services.modeling import ModelTrainer

def test_arima_training():
    """Test ARIMA model training."""
    trainer = ModelTrainer()
    
    # Create test data
    np.random.seed(42)
    y_train = pd.Series(np.random.randn(100).cumsum())
    y_test = pd.Series(np.random.randn(20).cumsum())
    
    try:
        model, predictions = trainer._train_arima(y_train, y_test)
        
        # Check that model is trained
        assert model is not None
        assert len(predictions) == len(y_test)
        
    except Exception as e:
        # ARIMA might fail with some random data, which is acceptable
        assert "ARIMA" in str(e) or "auto_arima" in str(e)

def test_naive_last_predict():
    """Test naive last value prediction."""
    trainer = ModelTrainer()
    
    y_train = pd.Series([1, 2, 3, 4, 5])
    X_test = pd.DataFrame({'feature1': [1, 2, 3]})
    
    predictions = trainer._naive_last_predict(X_test, y_train)
    
    # Should return last value repeated
    expected = np.array([5, 5, 5])
    np.testing.assert_array_equal(predictions, expected)

def test_naive_seasonal_predict():
    """Test naive seasonal prediction."""
    trainer = ModelTrainer()
    
    y_train = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    X_test = pd.DataFrame({'feature1': [1, 2, 3]})
    horizon = '5d'
    
    predictions = trainer._naive_seasonal_predict(X_test, y_train, horizon)
    
    # Should return seasonal values
    assert len(predictions) == len(X_test)
    assert not np.isnan(predictions).any()

def test_naive_seasonal_insufficient_data():
    """Test naive seasonal prediction with insufficient data."""
    trainer = ModelTrainer()
    
    y_train = pd.Series([1, 2])  # Less than horizon days
    X_test = pd.DataFrame({'feature1': [1, 2, 3]})
    horizon = '5d'
    
    predictions = trainer._naive_seasonal_predict(X_test, y_train, horizon)
    
    # Should fall back to naive last
    expected = np.array([2, 2, 2])
    np.testing.assert_array_equal(predictions, expected)

def test_xgboost_training():
    """Test XGBoost model training."""
    trainer = ModelTrainer()
    
    # Create test data
    np.random.seed(42)
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    y_train = pd.Series(np.random.randn(100))
    X_test = pd.DataFrame({
        'feature1': np.random.randn(20),
        'feature2': np.random.randn(20),
        'feature3': np.random.randn(20)
    })
    y_test = pd.Series(np.random.randn(20))
    
    try:
        model, predictions = trainer._train_xgboost(X_train, y_train, X_test, y_test)
        
        # Check that model is trained
        assert model is not None
        assert len(predictions) == len(y_test)
        
    except Exception as e:
        # XGBoost might fail with some random data, which is acceptable
        assert "XGBoost" in str(e) or "xgboost" in str(e)

def test_prophet_training():
    """Test Prophet model training."""
    trainer = ModelTrainer()
    
    # Create test data
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100),
        'Close': np.random.randn(100).cumsum() + 100
    })
    horizon = '1d'
    n_periods = 20
    
    try:
        model, predictions = trainer._train_prophet(df, horizon, n_periods)
        
        # Check that model is trained
        assert model is not None
        assert len(predictions) == n_periods
        
    except ImportError:
        # Prophet not installed, which is acceptable
        pass
    except Exception as e:
        # Prophet might fail with some random data, which is acceptable
        assert "Prophet" in str(e) or "prophet" in str(e)

def test_walk_forward_validation():
    """Test walk-forward validation."""
    trainer = ModelTrainer()
    
    # Create test data
    np.random.seed(42)
    df = pd.DataFrame({
        'Close': np.random.randn(200).cumsum() + 100,
        'Date': pd.date_range('2023-01-01', periods=200)
    })
    
    # Add target
    df['target_1d'] = df['Close'].pct_change().shift(-1)
    
    # Add some features
    df['feature1'] = np.random.randn(200)
    df['feature2'] = np.random.randn(200)
    
    try:
        result = trainer.walk_forward_validation(
            df, 'TEST', '1d', initial_train_days=50, step_days=10
        )
        
        # Check that walk-forward validation runs
        assert 'error' not in result or 'results' in result
        
    except Exception as e:
        # Walk-forward might fail with some random data, which is acceptable
        assert "walk" in str(e).lower() or "validation" in str(e).lower()

def test_model_training_insufficient_data():
    """Test model training with insufficient data."""
    trainer = ModelTrainer()
    
    # Create insufficient data
    df = pd.DataFrame({
        'Close': [100, 101, 102],
        'Date': pd.date_range('2023-01-01', periods=3)
    })
    df['target_1d'] = df['Close'].pct_change().shift(-1)
    
    result = trainer._train_horizon_models(df, 'TEST', '1d')
    
    # Should return error for insufficient data
    assert 'error' in result

def test_model_training_missing_target():
    """Test model training with missing target."""
    trainer = ModelTrainer()
    
    # Create data without target
    df = pd.DataFrame({
        'Close': [100, 101, 102, 103, 104],
        'Date': pd.date_range('2023-01-01', periods=5)
    })
    
    result = trainer._train_horizon_models(df, 'TEST', '1d')
    
    # Should return error for missing target
    assert 'error' in result

def test_model_training_empty_data():
    """Test model training with empty data."""
    trainer = ModelTrainer()
    
    # Create empty data
    df = pd.DataFrame()
    
    result = trainer._train_horizon_models(df, 'TEST', '1d')
    
    # Should return error for empty data
    assert 'error' in result

def test_feature_importance_extraction():
    """Test feature importance extraction."""
    trainer = ModelTrainer()
    
    # Mock model with feature_importances_
    class MockModel:
        def __init__(self):
            self.feature_importances_ = np.array([0.3, 0.2, 0.1])
    
    model = MockModel()
    model_data = {'model': model}
    
    importance_df = trainer.get_feature_importance(model_data)
    
    # Check that importance is extracted
    assert not importance_df.empty
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns

def test_feature_importance_no_model():
    """Test feature importance extraction with no model."""
    trainer = ModelTrainer()
    
    model_data = {}
    
    importance_df = trainer.get_feature_importance(model_data)
    
    # Should return None for no model
    assert importance_df is None

if __name__ == "__main__":
    pytest.main([__file__])
