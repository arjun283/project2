"""
Modeling service for commodity price prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

from utils.constants import HORIZONS, BACKTEST_PARAMS
from utils.seeds import set_seeds

class ModelTrainer:
    """Handles model training and prediction for commodity prices."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.set_seeds()
    
    def set_seeds(self):
        """Set random seeds for reproducibility."""
        set_seeds()
    
    def train_models(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        horizons: List[str] = None,
        use_prophet: bool = False
    ) -> Dict[str, Any]:
        """Train models for all horizons and return results."""
        
        if horizons is None:
            horizons = list(HORIZONS.keys())
        
        if df.empty or 'Close' not in df.columns:
            return {}
        
        results = {}
        
        for horizon in horizons:
            try:
                horizon_results = self._train_horizon_models(df, symbol, horizon, use_prophet)
                results[horizon] = horizon_results
            except Exception as e:
                print(f"Error training models for {symbol} {horizon}: {e}")
                results[horizon] = {'error': str(e)}
        
        return results
    
    def _train_horizon_models(self, df: pd.DataFrame, symbol: str, horizon: str, use_prophet: bool = False) -> Dict[str, Any]:
        """Train models for a specific horizon."""
        
        # Prepare data
        target_col = f'target_{horizon}'
        if target_col not in df.columns:
            return {'error': f'Target column {target_col} not found'}
        
        # Remove rows with missing targets
        df_clean = df.dropna(subset=[target_col]).copy()
        
        if len(df_clean) < 100:  # Need minimum data points
            return {'error': 'Insufficient data for training'}
        
        # Get feature columns
        feature_cols = [col for col in df_clean.columns if col not in [
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'data_quality',
            'target_1d', 'target_5d', 'target_20d', 'price_1d', 'price_5d', 'price_20d',
            'direction_1d', 'direction_5d', 'direction_20d', 'regime', 'regime_change'
        ]]
        
        X = df_clean[feature_cols].fillna(0)
        y = df_clean[target_col]
        
        # Remove rows with any NaN in features
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 50:
            return {'error': 'Insufficient clean data for training'}
        
        # Split data for walk-forward validation
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train models
        models = {}
        metrics = {}
        
        # 1. Naive models
        naive_last_pred = self._naive_last_predict(X_test, y_train)
        naive_seasonal_pred = self._naive_seasonal_predict(X_test, y_train, horizon)
        
        models['naive_last'] = {'predictions': naive_last_pred}
        models['naive_seasonal'] = {'predictions': naive_seasonal_pred}
        
        # 2. ARIMA
        try:
            arima_model, arima_pred = self._train_arima(y_train, y_test)
            models['arima'] = {'model': arima_model, 'predictions': arima_pred}
        except Exception as e:
            models['arima'] = {'error': str(e)}
        
        # 3. XGBoost
        try:
            xgb_model, xgb_pred = self._train_xgboost(X_train, y_train, X_test, y_test)
            models['xgboost'] = {'model': xgb_model, 'predictions': xgb_pred}
        except Exception as e:
            models['xgboost'] = {'error': str(e)}
        
        # 4. Prophet (optional)
        if use_prophet:
            try:
                prophet_model, prophet_pred = self._train_prophet(df_clean, horizon, len(y_test))
                models['prophet'] = {'model': prophet_model, 'predictions': prophet_pred}
            except Exception as e:
                models['prophet'] = {'error': str(e)}
        
        # Calculate metrics
        for model_name, model_data in models.items():
            if 'predictions' in model_data and 'error' not in model_data:
                pred = model_data['predictions']
                metrics[model_name] = self._calculate_metrics(y_test, pred)
            else:
                metrics[model_name] = {'error': model_data.get('error', 'Unknown error')}
        
        return {
            'models': models,
            'metrics': metrics,
            'feature_columns': feature_cols,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def _naive_last_predict(self, X_test: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        """Naive last value prediction."""
        return np.full(len(X_test), y_train.iloc[-1])
    
    def _naive_seasonal_predict(self, X_test: pd.DataFrame, y_train: pd.Series, horizon: str) -> np.ndarray:
        """Seasonal naive prediction."""
        horizon_days = HORIZONS[horizon]['days']
        
        if len(y_train) < horizon_days:
            return self._naive_last_predict(X_test, y_train)
        
        # Use the value from the same day of week/month as the target
        seasonal_values = []
        for i in range(len(X_test)):
            if i < len(y_train):
                seasonal_values.append(y_train.iloc[-(horizon_days + i)])
            else:
                seasonal_values.append(y_train.iloc[-1])
        
        return np.array(seasonal_values)
    
    def _train_arima(self, y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, np.ndarray]:
        """Train ARIMA model."""
        try:
            # Auto ARIMA with limited search space
            model = auto_arima(
                y_train,
                max_p=3, max_q=3, max_P=2, max_Q=2,
                seasonal=True,
                m=5,  # Weekly seasonality
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            # Make predictions
            predictions = model.predict(n_periods=len(y_test))
            
            return model, predictions
            
        except Exception as e:
            # Fallback to simple ARIMA
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(y_train, order=(1, 1, 1))
            fitted_model = model.fit()
            predictions = fitted_model.forecast(steps=len(y_test))
            
            return fitted_model, predictions
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, np.ndarray]:
        """Train XGBoost model."""
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        
        # Store scaler for later use
        self.scalers[f'xgboost_{X_train.columns.tolist()}'] = scaler
        
        return model, predictions
    
    def _train_prophet(self, df: pd.DataFrame, horizon: str, n_periods: int) -> Tuple[Any, np.ndarray]:
        """Train Prophet model (optional)."""
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            prophet_df = df[['Date', 'Close']].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df = prophet_df.dropna()
            
            if len(prophet_df) < 50:
                raise ValueError("Insufficient data for Prophet")
            
            # Train Prophet
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_df)
            
            # Make predictions
            future = model.make_future_dataframe(periods=n_periods)
            forecast = model.predict(future)
            
            # Extract predictions for the test period
            predictions = forecast['yhat'].iloc[-n_periods:].values
            
            return model, predictions
            
        except ImportError:
            raise ImportError("Prophet not installed. Install with: pip install prophet")
        except Exception as e:
            raise Exception(f"Prophet training failed: {e}")
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction metrics."""
        
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {'error': 'No valid predictions'}
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # MAPE
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
        
        # Directional accuracy
        direction_accuracy = np.mean(np.sign(y_true_clean) == np.sign(y_pred_clean)) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'n_predictions': len(y_true_clean)
        }
    
    def predict(self, model_name: str, model_data: Dict, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model."""
        
        if 'error' in model_data:
            raise ValueError(f"Model error: {model_data['error']}")
        
        if model_name == 'naive_last':
            return np.full(len(X), model_data['predictions'][0])
        
        elif model_name == 'naive_seasonal':
            return model_data['predictions']
        
        elif model_name == 'arima':
            model = model_data['model']
            return model.predict(n_periods=len(X))
        
        elif model_name == 'xgboost':
            model = model_data['model']
            # Scale features if scaler exists
            scaler_key = f'xgboost_{X.columns.tolist()}'
            if scaler_key in self.scalers:
                X_scaled = self.scalers[scaler_key].transform(X)
            else:
                X_scaled = X.values
            return model.predict(X_scaled)
        
        elif model_name == 'prophet':
            # Prophet predictions are more complex, return cached predictions
            return model_data['predictions']
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def get_feature_importance(self, model_data: Dict) -> Optional[pd.DataFrame]:
        """Get feature importance from XGBoost model."""
        
        if 'model' not in model_data or not hasattr(model_data['model'], 'feature_importances_'):
            return None
        
        model = model_data['model']
        feature_names = getattr(model, 'feature_names_in_', [f'feature_{i}' for i in range(len(model.feature_importances_))])
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def walk_forward_validation(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        horizon: str,
        initial_train_days: int = 500,
        step_days: int = 5
    ) -> Dict[str, Any]:
        """Perform walk-forward validation."""
        
        if df.empty or f'target_{horizon}' not in df.columns:
            return {'error': 'Invalid data or missing target'}
        
        # Prepare data
        df_clean = df.dropna(subset=[f'target_{horizon}']).copy()
        
        if len(df_clean) < initial_train_days + 100:
            return {'error': 'Insufficient data for walk-forward validation'}
        
        # Get feature columns
        feature_cols = [col for col in df_clean.columns if col not in [
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'data_quality',
            'target_1d', 'target_5d', 'target_20d', 'price_1d', 'price_5d', 'price_20d',
            'direction_1d', 'direction_5d', 'direction_20d', 'regime', 'regime_change'
        ]]
        
        # Walk-forward validation
        results = []
        start_idx = initial_train_days
        
        while start_idx + step_days < len(df_clean):
            # Training data
            train_end = start_idx
            X_train = df_clean.iloc[:train_end][feature_cols].fillna(0)
            y_train = df_clean.iloc[:train_end][f'target_{horizon}']
            
            # Test data
            test_start = start_idx
            test_end = min(start_idx + step_days, len(df_clean))
            X_test = df_clean.iloc[test_start:test_end][feature_cols].fillna(0)
            y_test = df_clean.iloc[test_start:test_end][f'target_{horizon}']
            
            # Remove NaN values
            train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
            test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
            
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask]
            X_test_clean = X_test[test_mask]
            y_test_clean = y_test[test_mask]
            
            if len(X_train_clean) < 50 or len(X_test_clean) == 0:
                start_idx += step_days
                continue
            
            # Train XGBoost model
            try:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_clean)
                X_test_scaled = scaler.transform(X_test_clean)
                
                model = xgb.XGBRegressor(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42
                )
                
                model.fit(X_train_scaled, y_train_clean)
                predictions = model.predict(X_test_scaled)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test_clean, predictions)
                
                results.append({
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'metrics': metrics,
                    'predictions': predictions.tolist(),
                    'actual': y_test_clean.tolist()
                })
                
            except Exception as e:
                print(f"Error in walk-forward step {start_idx}: {e}")
            
            start_idx += step_days
        
        # Aggregate results
        if not results:
            return {'error': 'No valid walk-forward steps'}
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in ['rmse', 'mae', 'r2', 'mape', 'direction_accuracy']:
            values = [r['metrics'].get(metric, np.nan) for r in results if metric in r['metrics']]
            if values:
                avg_metrics[metric] = np.nanmean(values)
        
        return {
            'results': results,
            'avg_metrics': avg_metrics,
            'n_steps': len(results),
            'feature_columns': feature_cols
        }
