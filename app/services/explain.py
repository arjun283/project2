"""
Model explainability and SHAP analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from utils.constants import TECHNICAL_PARAMS

class ModelExplainer:
    """Handles model explainability and SHAP analysis."""
    
    def __init__(self):
        self.explainers = {}
        self.shap_values = {}
        self.feature_names = []
    
    def explain_model(
        self,
        model,
        X: pd.DataFrame,
        model_name: str = 'xgboost',
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """Generate model explanations using SHAP."""
        
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not available. Install with: pip install shap'}
        
        if X.empty:
            return {'error': 'No data provided'}
        
        # Limit samples for performance
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
        else:
            X_sample = X.copy()
        
        try:
            # Create explainer based on model type
            if model_name == 'xgboost':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            elif model_name == 'linear':
                explainer = shap.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_sample)
            else:
                # Use KernelExplainer as fallback
                explainer = shap.KernelExplainer(model.predict, X_sample.iloc[:50])
                shap_values = explainer.shap_values(X_sample.iloc[:50])
            
            # Store for later use
            self.explainers[model_name] = explainer
            self.shap_values[model_name] = shap_values
            self.feature_names = X_sample.columns.tolist()
            
            # Generate explanations
            explanations = self._generate_explanations(explainer, shap_values, X_sample)
            
            return explanations
            
        except Exception as e:
            return {'error': f'SHAP analysis failed: {str(e)}'}
    
    def _generate_explanations(
        self, 
        explainer, 
        shap_values: np.ndarray, 
        X: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate various types of explanations."""
        
        explanations = {}
        
        # Feature importance (mean absolute SHAP values)
        if len(shap_values.shape) == 2:  # 2D array
            feature_importance = np.abs(shap_values).mean(axis=0)
        else:  # 1D array
            feature_importance = np.abs(shap_values)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        explanations['feature_importance'] = importance_df
        
        # Summary plot data
        explanations['summary_data'] = {
            'shap_values': shap_values.tolist() if len(shap_values.shape) == 2 else shap_values.tolist(),
            'feature_values': X.values.tolist(),
            'feature_names': X.columns.tolist()
        }
        
        # Waterfall plot for latest prediction
        if len(shap_values.shape) == 2 and len(shap_values) > 0:
            latest_shap = shap_values[-1]
            latest_values = X.iloc[-1].values
            
            # Calculate base value (expected value)
            base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
            
            # Sort by absolute SHAP value
            sorted_indices = np.argsort(np.abs(latest_shap))[::-1]
            
            waterfall_data = []
            cumulative_value = base_value
            
            for idx in sorted_indices[:10]:  # Top 10 features
                feature_name = X.columns[idx]
                shap_value = latest_shap[idx]
                feature_value = latest_values[idx]
                
                waterfall_data.append({
                    'feature': feature_name,
                    'shap_value': shap_value,
                    'feature_value': feature_value,
                    'cumulative_value': cumulative_value + shap_value
                })
                
                cumulative_value += shap_value
            
            explanations['waterfall_data'] = {
                'base_value': base_value,
                'final_value': cumulative_value,
                'steps': waterfall_data
            }
        
        return explanations
    
    def get_feature_importance_plot_data(self, model_name: str = 'xgboost') -> Dict[str, Any]:
        """Get data for feature importance plot."""
        
        if model_name not in self.explainers:
            return {'error': 'No explainer found for model'}
        
        explainer = self.explainers[model_name]
        shap_values = self.shap_values[model_name]
        
        if len(shap_values.shape) == 2:
            feature_importance = np.abs(shap_values).mean(axis=0)
        else:
            feature_importance = np.abs(shap_values)
        
        # Sort features by importance
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        plot_data = {
            'features': [self.feature_names[i] for i in sorted_indices],
            'importance': [feature_importance[i] for i in sorted_indices],
            'colors': ['red' if x > 0 else 'blue' for x in feature_importance[sorted_indices]]
        }
        
        return plot_data
    
    def get_summary_plot_data(self, model_name: str = 'xgboost') -> Dict[str, Any]:
        """Get data for SHAP summary plot."""
        
        if model_name not in self.shap_values:
            return {'error': 'No SHAP values found for model'}
        
        shap_values = self.shap_values[model_name]
        
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        
        return {
            'shap_values': shap_values.tolist(),
            'feature_names': self.feature_names,
            'max_display': min(20, len(self.feature_names))
        }
    
    def get_waterfall_plot_data(self, model_name: str = 'xgboost', sample_idx: int = -1) -> Dict[str, Any]:
        """Get data for SHAP waterfall plot."""
        
        if model_name not in self.explainers or model_name not in self.shap_values:
            return {'error': 'No explainer or SHAP values found for model'}
        
        explainer = self.explainers[model_name]
        shap_values = self.shap_values[model_name]
        
        if len(shap_values.shape) == 2:
            sample_shap = shap_values[sample_idx]
        else:
            sample_shap = shap_values
        
        # Calculate base value
        base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
        
        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(sample_shap))[::-1]
        
        waterfall_data = []
        cumulative_value = base_value
        
        for idx in sorted_indices[:15]:  # Top 15 features
            feature_name = self.feature_names[idx]
            shap_value = sample_shap[idx]
            
            waterfall_data.append({
                'feature': feature_name,
                'shap_value': shap_value,
                'cumulative_value': cumulative_value + shap_value
            })
            
            cumulative_value += shap_value
        
        return {
            'base_value': base_value,
            'final_value': cumulative_value,
            'steps': waterfall_data
        }
    
    def get_partial_dependence_data(
        self, 
        model, 
        X: pd.DataFrame, 
        feature_name: str,
        model_name: str = 'xgboost'
    ) -> Dict[str, Any]:
        """Get partial dependence data for a feature."""
        
        if feature_name not in X.columns:
            return {'error': f'Feature {feature_name} not found'}
        
        # Create grid of feature values
        feature_values = X[feature_name].dropna()
        if len(feature_values) == 0:
            return {'error': 'No valid feature values'}
        
        # Create grid
        min_val = feature_values.min()
        max_val = feature_values.max()
        grid = np.linspace(min_val, max_val, 50)
        
        # Calculate partial dependence
        partial_dependence = []
        
        for val in grid:
            # Create modified dataset
            X_modified = X.copy()
            X_modified[feature_name] = val
            
            # Make predictions
            if hasattr(model, 'predict'):
                pred = model.predict(X_modified)
                partial_dependence.append(pred.mean())
            else:
                partial_dependence.append(0)
        
        return {
            'feature_values': grid.tolist(),
            'partial_dependence': partial_dependence,
            'feature_name': feature_name
        }
    
    def get_feature_interaction_data(
        self, 
        model_name: str = 'xgboost',
        top_n: int = 10
    ) -> Dict[str, Any]:
        """Get feature interaction data."""
        
        if model_name not in self.shap_values:
            return {'error': 'No SHAP values found for model'}
        
        shap_values = self.shap_values[model_name]
        
        if len(shap_values.shape) == 1:
            return {'error': 'Need multiple samples for interaction analysis'}
        
        # Calculate feature interactions (simplified)
        interactions = []
        
        for i in range(min(top_n, len(self.feature_names))):
            for j in range(i + 1, min(top_n, len(self.feature_names))):
                # Calculate correlation between SHAP values
                corr = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]
                
                interactions.append({
                    'feature1': self.feature_names[i],
                    'feature2': self.feature_names[j],
                    'interaction_strength': abs(corr)
                })
        
        # Sort by interaction strength
        interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
        
        return {
            'interactions': interactions[:20],  # Top 20 interactions
            'feature_names': self.feature_names
        }
    
    def get_model_diagnostics(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Get model diagnostics and residual analysis."""
        
        if X.empty or y.empty:
            return {'error': 'No data provided'}
        
        try:
            # Make predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(X)
            else:
                return {'error': 'Model does not support prediction'}
            
            # Calculate residuals
            residuals = y - predictions
            
            # Basic diagnostics
            diagnostics = {
                'n_samples': len(y),
                'n_features': len(X.columns),
                'mean_residual': residuals.mean(),
                'std_residual': residuals.std(),
                'residual_skewness': residuals.skew(),
                'residual_kurtosis': residuals.kurtosis(),
                'r_squared': 1 - (residuals.var() / y.var()) if y.var() > 0 else 0
            }
            
            # Residual analysis
            residual_analysis = {
                'residuals': residuals.tolist(),
                'predictions': predictions.tolist(),
                'actual': y.tolist(),
                'residual_quantiles': residuals.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
            }
            
            return {
                'diagnostics': diagnostics,
                'residual_analysis': residual_analysis
            }
            
        except Exception as e:
            return {'error': f'Diagnostics failed: {str(e)}'}
    
    def get_feature_correlation_matrix(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Get feature correlation matrix."""
        
        if X.empty:
            return {'error': 'No data provided'}
        
        # Calculate correlation matrix
        corr_matrix = X.corr()
        
        # Convert to format suitable for plotting
        correlation_data = {
            'features': corr_matrix.columns.tolist(),
            'correlation_matrix': corr_matrix.values.tolist(),
            'max_correlation': corr_matrix.abs().max().max()
        }
        
        return correlation_data
    
    def get_feature_statistics(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get comprehensive feature statistics."""
        
        if X.empty:
            return pd.DataFrame()
        
        stats = []
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                stats.append({
                    'feature': col,
                    'count': X[col].count(),
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'min': X[col].min(),
                    'max': X[col].max(),
                    'skewness': X[col].skew(),
                    'kurtosis': X[col].kurtosis(),
                    'missing_pct': (X[col].isnull().sum() / len(X)) * 100
                })
        
        return pd.DataFrame(stats)
