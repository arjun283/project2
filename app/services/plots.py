"""
Plotting utilities for the Commodity Price Predictor app.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st

class PlotBuilder:
    """Creates interactive plots for the app."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set1
    
    def create_price_chart(
        self, 
        df: pd.DataFrame, 
        symbol: str,
        title: str = None,
        show_volume: bool = True
    ) -> go.Figure:
        """Create interactive price chart with volume."""
        
        if df.empty or 'Close' not in df.columns:
            return go.Figure()
        
        if title is None:
            title = f"{symbol} Price Chart"
        
        # Create subplots
        if show_volume and 'Volume' in df.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(title, 'Volume'),
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        # Price candlestick chart
        if 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns:
            candlestick = go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            )
        else:
            # Line chart if OHLC not available
            candlestick = go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=2)
            )
        
        if show_volume and 'Volume' in df.columns:
            fig.add_trace(candlestick, row=1, col=1)
            
            # Volume chart
            volume_colors = ['#26a69a' if close >= open else '#ef5350' 
                           for close, open in zip(df['Close'], df['Open'])]
            
            volume_bar = go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7
            )
            fig.add_trace(volume_bar, row=2, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Add moving averages
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1, dash='dash')
                ),
                row=1 if show_volume and 'Volume' in df.columns else None,
                col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            showlegend=True,
            height=600 if show_volume and 'Volume' in df.columns else 400,
            hovermode='x unified'
        )
        
        # Update x-axis
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        return fig
    
    def create_returns_chart(
        self, 
        df: pd.DataFrame, 
        symbol: str,
        title: str = None
    ) -> go.Figure:
        """Create returns distribution chart."""
        
        if df.empty or 'returns' not in df.columns:
            return go.Figure()
        
        if title is None:
            title = f"{symbol} Returns Distribution"
        
        returns = df['returns'].dropna()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Returns Time Series', 'Returns Distribution', 
                          'Q-Q Plot', 'Rolling Volatility'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Returns time series
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=returns,
                mode='lines',
                name='Returns',
                line=dict(color='#1f77b4', width=1)
            ),
            row=1, col=1
        )
        
        # Returns histogram
        fig.add_trace(
            go.Histogram(
                x=returns,
                name='Returns Distribution',
                nbinsx=50,
                marker_color='#1f77b4',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Q-Q plot (simplified)
        from scipy import stats
        qq_data = stats.probplot(returns, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='#1f77b4', size=4)
            ),
            row=2, col=1
        )
        
        # Rolling volatility
        if 'vol_20d' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['vol_20d'],
                    mode='lines',
                    name='20d Volatility',
                    line=dict(color='red', width=2)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_correlation_heatmap(
        self, 
        data: Dict[str, pd.DataFrame],
        title: str = "Commodity Returns Correlation"
    ) -> go.Figure:
        """Create correlation heatmap for commodities."""
        
        # Prepare correlation data
        returns_data = {}
        for symbol, df in data.items():
            if 'returns' in df.columns:
                returns_data[symbol] = df['returns'].dropna()
        
        if not returns_data:
            return go.Figure()
        
        # Create correlation matrix
        returns_df = pd.DataFrame(returns_data)
        corr_matrix = returns_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=500,
            width=500
        )
        
        return fig
    
    def create_regime_chart(
        self, 
        df: pd.DataFrame, 
        symbol: str,
        title: str = None
    ) -> go.Figure:
        """Create regime visualization chart."""
        
        if df.empty or 'regime' not in df.columns or 'Close' not in df.columns:
            return go.Figure()
        
        if title is None:
            title = f"{symbol} Price with Regime Detection"
        
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='black', width=2)
            )
        )
        
        # Add regime background colors
        regime_colors = {
            'bull_quiet': 'rgba(46, 139, 87, 0.3)',
            'bull_volatile': 'rgba(50, 205, 50, 0.3)',
            'bear_quiet': 'rgba(220, 20, 60, 0.3)',
            'bear_volatile': 'rgba(255, 99, 71, 0.3)',
            'sideways_quiet': 'rgba(211, 211, 211, 0.3)',
            'sideways_volatile': 'rgba(169, 169, 169, 0.3)'
        }
        
        # Add regime rectangles
        current_regime = None
        start_date = None
        
        for _, row in df.iterrows():
            if row['regime'] != current_regime:
                if current_regime is not None and start_date is not None:
                    fig.add_vrect(
                        x0=start_date,
                        x1=row['Date'],
                        fillcolor=regime_colors.get(current_regime, 'rgba(0,0,0,0.1)'),
                        layer="below",
                        line_width=0,
                        annotation_text=current_regime.replace('_', ' ').title(),
                        annotation_position="top left"
                    )
                
                current_regime = row['regime']
                start_date = row['Date']
        
        # Add last regime
        if current_regime is not None and start_date is not None:
            fig.add_vrect(
                x0=start_date,
                x1=df['Date'].iloc[-1],
                fillcolor=regime_colors.get(current_regime, 'rgba(0,0,0,0.1)'),
                layer="below",
                line_width=0,
                annotation_text=current_regime.replace('_', ' ').title(),
                annotation_position="top left"
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_performance_chart(
        self, 
        returns: pd.Series, 
        title: str = "Strategy Performance",
        initial_capital: float = 10000
    ) -> go.Figure:
        """Create performance chart with equity curve and drawdown."""
        
        if returns.empty:
            return go.Figure()
        
        # Calculate equity curve
        equity_curve = (1 + returns).cumprod() * initial_capital
        
        # Calculate drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Equity Curve', 'Drawdown'),
            row_heights=[0.7, 0.3]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=equity_curve,
                mode='lines',
                name='Equity Curve',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    def create_feature_importance_chart(
        self, 
        importance_data: pd.DataFrame,
        title: str = "Feature Importance",
        top_n: int = 20
    ) -> go.Figure:
        """Create feature importance chart."""
        
        if importance_data.empty:
            return go.Figure()
        
        # Get top N features
        top_features = importance_data.head(top_n)
        
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                y=top_features['feature'],
                x=top_features['importance'],
                orientation='h',
                marker_color='#1f77b4',
                text=top_features['importance'].round(3),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_white',
            height=max(400, len(top_features) * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_shap_summary_chart(
        self, 
        shap_data: Dict[str, Any],
        title: str = "SHAP Summary"
    ) -> go.Figure:
        """Create SHAP summary plot."""
        
        if 'error' in shap_data:
            return go.Figure()
        
        shap_values = np.array(shap_data['shap_values'])
        feature_names = shap_data['feature_names']
        
        if len(shap_values.shape) != 2:
            return go.Figure()
        
        # Create scatter plot
        fig = go.Figure()
        
        for i, feature in enumerate(feature_names[:20]):  # Top 20 features
            fig.add_trace(
                go.Scatter(
                    x=shap_values[:, i],
                    y=[feature] * len(shap_values),
                    mode='markers',
                    name=feature,
                    marker=dict(
                        size=6,
                        color=shap_values[:, i],
                        colorscale='RdBu',
                        showscale=False,
                        opacity=0.7
                    ),
                    showlegend=False
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='SHAP Value',
            yaxis_title='Feature',
            template='plotly_white',
            height=max(400, len(feature_names) * 20)
        )
        
        return fig
    
    def create_waterfall_chart(
        self, 
        waterfall_data: Dict[str, Any],
        title: str = "SHAP Waterfall"
    ) -> go.Figure:
        """Create SHAP waterfall chart."""
        
        if 'error' in waterfall_data or 'steps' not in waterfall_data:
            return go.Figure()
        
        steps = waterfall_data['steps']
        base_value = waterfall_data['base_value']
        final_value = waterfall_data['final_value']
        
        # Prepare data for waterfall
        x_labels = ['Base'] + [step['feature'] for step in steps] + ['Final']
        y_values = [base_value] + [step['cumulative_value'] for step in steps] + [final_value]
        
        # Calculate differences
        diffs = [0] + [step['shap_value'] for step in steps] + [0]
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="SHAP Values",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(steps) + ["total"],
            x=x_labels,
            y=diffs,
            textposition="outside",
            text=[f"{diff:.3f}" for diff in diffs],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_metrics_table(
        self, 
        metrics: Dict[str, Any],
        title: str = "Performance Metrics"
    ) -> go.Figure:
        """Create metrics table."""
        
        if not metrics or 'error' in metrics:
            return go.Figure()
        
        # Prepare table data
        metric_names = []
        metric_values = []
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_names.append(key.replace('_', ' ').title())
                if isinstance(value, float):
                    metric_values.append(f"{value:.4f}")
                else:
                    metric_values.append(f"{value}")
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='#1f77b4',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[metric_names, metric_values],
                fill_color='white',
                font=dict(size=11),
                align='left'
            )
        )])
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=min(600, len(metric_names) * 30 + 100)
        )
        
        return fig
