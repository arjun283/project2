"""
Test metrics calculations.
"""

import pytest
import numpy as np
import pandas as pd
from services.modeling import ModelTrainer

def test_rmse_calculation():
    """Test RMSE calculation."""
    trainer = ModelTrainer()
    
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    metrics = trainer._calculate_metrics(pd.Series(y_true), y_pred)
    
    expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert abs(metrics['rmse'] - expected_rmse) < 1e-10

def test_mae_calculation():
    """Test MAE calculation."""
    trainer = ModelTrainer()
    
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    metrics = trainer._calculate_metrics(pd.Series(y_true), y_pred)
    
    expected_mae = np.mean(np.abs(y_true - y_pred))
    assert abs(metrics['mae'] - expected_mae) < 1e-10

def test_r2_calculation():
    """Test R² calculation."""
    trainer = ModelTrainer()
    
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    metrics = trainer._calculate_metrics(pd.Series(y_true), y_pred)
    
    # R² should be close to 1 for good predictions
    assert metrics['r2'] > 0.9

def test_mape_calculation():
    """Test MAPE calculation."""
    trainer = ModelTrainer()
    
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    metrics = trainer._calculate_metrics(pd.Series(y_true), y_pred)
    
    expected_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    assert abs(metrics['mape'] - expected_mape) < 1e-10

def test_directional_accuracy():
    """Test directional accuracy calculation."""
    trainer = ModelTrainer()
    
    y_true = np.array([1, -1, 1, -1, 1])
    y_pred = np.array([0.5, -0.5, 0.5, -0.5, 0.5])
    
    metrics = trainer._calculate_metrics(pd.Series(y_true), y_pred)
    
    # All predictions should have correct direction
    assert metrics['direction_accuracy'] == 100.0

def test_empty_inputs():
    """Test handling of empty inputs."""
    trainer = ModelTrainer()
    
    y_true = pd.Series([])
    y_pred = np.array([])
    
    metrics = trainer._calculate_metrics(y_true, y_pred)
    
    assert 'error' in metrics

def test_nan_inputs():
    """Test handling of NaN inputs."""
    trainer = ModelTrainer()
    
    y_true = pd.Series([1, 2, np.nan, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    metrics = trainer._calculate_metrics(y_true, y_pred)
    
    # Should handle NaN values gracefully
    assert 'rmse' in metrics
    assert not np.isnan(metrics['rmse'])

if __name__ == "__main__":
    pytest.main([__file__])
