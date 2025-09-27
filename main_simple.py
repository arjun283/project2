# Force commit: updated by GitHub Copilot on 2025-09-27
"""
Commodity Price Predictor - With Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Commodity Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Commodity symbols
COMMODITIES = {
    'CL=F': {'name': 'WTI Crude Oil', 'category': 'Energy'},
    'BZ=F': {'name': 'Brent Crude Oil', 'category': 'Energy'},
    'NG=F': {'name': 'Natural Gas', 'category': 'Energy'},
    'GC=F': {'name': 'Gold', 'category': 'Metals'},
    'SI=F': {'name': 'Silver', 'category': 'Metals'},
    'HG=F': {'name': 'Copper', 'category': 'Metals'},
    'ZC=F': {'name': 'Corn', 'category': 'Agriculture'},
    'ZS=F': {'name': 'Soybeans', 'category': 'Agriculture'},
}

def load_data(symbol, days=365):
    """Load commodity data from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{days}d")
        return data
    except Exception as e:
        st.error(f"Error loading {symbol}: {str(e)}")
        return pd.DataFrame()

def create_features(df):
    """Create technical features for prediction."""
    df = df.copy()
    
    # Price features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Price ratios
    df['Price_MA5_Ratio'] = df['Close'] / df['MA_5']
    df['Price_MA20_Ratio'] = df['Close'] / df['MA_20']
    
    # Volatility
    df['Volatility_5'] = df['Returns'].rolling(window=5).std()
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Volume features
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
    
    return df

def train_model(df, target_col='Close', test_size=0.2):
    """Train a Random Forest model for price prediction."""
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Returns', 'Log_Returns']]
    feature_cols = [col for col in feature_cols if not col.startswith('MA_') or col in ['MA_5', 'MA_20']]
    
    # Remove rows with NaN values
    df_clean = df[feature_cols + [target_col]].dropna()
    
    if len(df_clean) < 50:
        return None, None, None, None
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Split data
    split_idx = int(len(df_clean) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, feature_cols, {
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }

def make_predictions(model, scaler, feature_cols, df, n_days=30):
    """Make future predictions."""
    if model is None:
        return None
    
    # Get the last available data
    last_data = df[feature_cols].iloc[-1:].fillna(method='ffill')
    
    predictions = []
    current_data = last_data.copy()
    
    for _ in range(n_days):
        # Scale the data
        scaled_data = scaler.transform(current_data)
        
        # Make prediction
        pred = model.predict(scaled_data)[0]
        predictions.append(pred)
        
        # Update features for next prediction (simplified)
        current_data = current_data.copy()
        current_data['Close_Lag_1'] = pred
        current_data['Close_Lag_2'] = current_data['Close_Lag_1'].shift(1)
        current_data['Close_Lag_3'] = current_data['Close_Lag_2'].shift(1)
        
        # Update moving averages (simplified)
        if 'MA_5' in current_data.columns:
            current_data['MA_5'] = pred
        if 'MA_20' in current_data.columns:
            current_data['MA_20'] = pred
    
    return predictions

def main():
    """Main application function."""
    
    st.title("📈 Commodity Price Predictor")
    st.markdown("Welcome to the Commodity Price Predictor - your comprehensive tool for commodity market analysis.")
    
    # Sidebar
    st.sidebar.title("📊 Data Controls")
    
    # Symbol selection
    selected_symbols = st.sidebar.multiselect(
        "Select Commodities",
        list(COMMODITIES.keys()),
        default=list(COMMODITIES.keys())[:3]
    )
    
    # Date range
    days_back = st.sidebar.slider("Days Back", 30, 365, 90)
    
    # Load data button
    if st.sidebar.button("🔄 Load Data"):
        with st.spinner("Loading data..."):
            data_dict = {}
            for symbol in selected_symbols:
                data = load_data(symbol, days_back)
                if not data.empty:
                    # Create features for prediction
                    data_with_features = create_features(data)
                    data_dict[symbol] = data_with_features
            st.session_state.data = data_dict
            st.success(f"Loaded data for {len(data_dict)} commodities")
    
    # Train models button
    if st.sidebar.button("🤖 Train Models"):
        if 'data' in st.session_state and st.session_state.data:
            with st.spinner("Training models..."):
                models = {}
                for symbol, df in st.session_state.data.items():
                    if not df.empty and len(df) > 50:
                        model, scaler, features, metrics = train_model(df)
                        if model is not None:
                            models[symbol] = {
                                'model': model,
                                'scaler': scaler,
                                'features': features,
                                'metrics': metrics
                            }
                st.session_state.models = models
                st.success(f"Trained models for {len(models)} commodities")
        else:
            st.warning("Please load data first!")
    
    # Check if data is loaded
    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data first using the sidebar controls.")
        return
    
    data = st.session_state.data
    
    # Key Metrics
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Commodities", len(data))
    
    with col2:
        total_data_points = sum(len(df) for df in data.values())
        st.metric("Total Data Points", f"{total_data_points:,}")
    
    with col3:
        avg_vol = np.mean([df['Close'].pct_change().std() * np.sqrt(252) 
                          for df in data.values() if 'Close' in df.columns])
        st.metric("Avg Volatility", f"{avg_vol:.1%}")
    
    with col4:
        st.metric("Last Update", datetime.now().strftime("%Y-%m-%d"))
    
    # Market Overview
    st.subheader("🌍 Market Overview")
    
    # Create summary table
    summary_data = []
    for symbol, df in data.items():
        if df.empty or 'Close' not in df.columns:
            continue
        
        # Calculate metrics
        last_close = df['Close'].iloc[-1]
        returns_20d = df['Close'].pct_change(20).iloc[-1]
        vol_20d = df['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        
        summary_data.append({
            'Symbol': symbol,
            'Name': COMMODITIES[symbol]['name'],
            'Category': COMMODITIES[symbol]['category'],
            'Last Close': f"${last_close:.2f}",
            '20d Return': f"{returns_20d:.1%}",
            '20d Vol': f"{vol_20d:.1%}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Price Charts with Predictions
    st.subheader("📈 Price Charts & Predictions")
    
    # Select commodities to display
    chart_symbols = st.multiselect(
        "Select commodities to display",
        list(data.keys()),
        default=list(data.keys())[:3]
    )
    
    if chart_symbols:
        # Create tabs for each commodity
        tabs = st.tabs(chart_symbols)
        
        for i, symbol in enumerate(chart_symbols):
            with tabs[i]:
                if symbol in data and not data[symbol].empty:
                    df = data[symbol]
                    
                    # Create price chart with predictions
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Add predictions if model exists
                    if 'models' in st.session_state and symbol in st.session_state.models:
                        model_info = st.session_state.models[symbol]
                        predictions = make_predictions(
                            model_info['model'],
                            model_info['scaler'],
                            model_info['features'],
                            df,
                            n_days=30
                        )
                        
                        if predictions:
                            # Create future dates
                            last_date = df.index[-1]
                            future_dates = pd.date_range(
                                start=last_date + timedelta(days=1),
                                periods=len(predictions),
                                freq='D'
                            )
                            
                            # Add prediction line
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=predictions,
                                mode='lines',
                                name='Predicted Price',
                                line=dict(color='#ff7f0e', width=2, dash='dash')
                            ))
                            
                            # Add confidence interval (simplified)
                            upper_bound = [p * 1.05 for p in predictions]
                            lower_bound = [p * 0.95 for p in predictions]
                            
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=upper_bound,
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=lower_bound,
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(255, 127, 14, 0.2)',
                                name='Confidence Interval'
                            ))
                    
                    fig.update_layout(
                        title=f"{symbol} Price Chart with Predictions",
                        xaxis_title='Date',
                        yaxis_title='Price',
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Price statistics and predictions
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                    with col2:
                        returns_1d = df['Close'].pct_change().iloc[-1]
                        st.metric("1d Return", f"{returns_1d:.2%}")
                    with col3:
                        returns_20d = df['Close'].pct_change(20).iloc[-1]
                        st.metric("20d Return", f"{returns_20d:.2%}")
                    with col4:
                        vol_20d = df['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
                        st.metric("20d Volatility", f"{vol_20d:.1%}")
                    
                    # Show predictions if available
                    if 'models' in st.session_state and symbol in st.session_state.models:
                        st.subheader("🔮 Predictions")
                        
                        model_info = st.session_state.models[symbol]
                        predictions = make_predictions(
                            model_info['model'],
                            model_info['scaler'],
                            model_info['features'],
                            df,
                            n_days=30
                        )
                        
                        if predictions:
                            # Prediction metrics
                            pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
                            
                            with pred_col1:
                                next_price = predictions[0]
                                current_price = df['Close'].iloc[-1]
                                change = next_price - current_price
                                change_pct = change / current_price
                                st.metric(
                                    "Next Day Prediction", 
                                    f"${next_price:.2f}",
                                    f"{change_pct:.2%}"
                                )
                            
                            with pred_col2:
                                week_price = predictions[6] if len(predictions) > 6 else predictions[-1]
                                week_change = week_price - current_price
                                week_change_pct = week_change / current_price
                                st.metric(
                                    "1 Week Prediction", 
                                    f"${week_price:.2f}",
                                    f"{week_change_pct:.2%}"
                                )
                            
                            with pred_col3:
                                month_price = predictions[29] if len(predictions) > 29 else predictions[-1]
                                month_change = month_price - current_price
                                month_change_pct = month_change / current_price
                                st.metric(
                                    "1 Month Prediction", 
                                    f"${month_price:.2f}",
                                    f"{month_change_pct:.2%}"
                                )
                            
                            with pred_col4:
                                st.metric("Model R²", f"{model_info['metrics']['r2']:.3f}")
                            
                            # Model performance
                            st.subheader("📊 Model Performance")
                            perf_col1, perf_col2, perf_col3 = st.columns(3)
                            
                            with perf_col1:
                                st.metric("Mean Absolute Error", f"${model_info['metrics']['mae']:.2f}")
                            with perf_col2:
                                st.metric("Mean Squared Error", f"${model_info['metrics']['mse']:.2f}")
                            with perf_col3:
                                st.metric("R² Score", f"{model_info['metrics']['r2']:.3f}")
                            
                            # Feature importance
                            if hasattr(model_info['model'], 'feature_importances_'):
                                st.subheader("🎯 Feature Importance")
                                feature_importance = pd.DataFrame({
                                    'Feature': model_info['features'],
                                    'Importance': model_info['model'].feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig_importance = go.Figure(go.Bar(
                                    x=feature_importance['Importance'],
                                    y=feature_importance['Feature'],
                                    orientation='h'
                                ))
                                fig_importance.update_layout(
                                    title="Feature Importance",
                                    xaxis_title="Importance",
                                    height=400
                                )
                                st.plotly_chart(fig_importance, use_container_width=True)
    
    # Data Export
    st.subheader("💾 Data Export")
    
    if st.button("📥 Download All Data as CSV"):
        # Combine all data
        all_data = []
        for symbol, df in data.items():
            df_copy = df.copy()
            df_copy['Symbol'] = symbol
            df_copy = df_copy.reset_index()
            all_data.append(df_copy)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"commodity_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("**Commodity Price Predictor** - Built with Streamlit")

if __name__ == "__main__":
    main()
