# 📈 Commodity Price Predictor

A comprehensive Streamlit application for commodity price prediction, analysis, and trading strategy backtesting. Built with Python, Streamlit, and machine learning libraries.

## 🚀 Features

### 📊 Data Management
- **Multi-source data loading**: yfinance, FRED API, cached CSV fallbacks
- **Real-time data refresh**: Configurable caching and update frequencies
- **Data quality monitoring**: Gap detection, missing data analysis
- **Cross-asset analysis**: Correlation matrices and macro indicators

### 🤖 Machine Learning Models
- **Multiple model types**: Naive, ARIMA/SARIMAX, XGBoost, Prophet (optional)
- **Walk-forward validation**: Proper time series validation without look-ahead bias
- **Multi-horizon predictions**: 1-day, 5-day, and 20-day forecasts
- **Model comparison**: Comprehensive metrics and performance analysis

### 📈 Trading & Backtesting
- **Strategy simulation**: Customizable trading rules and risk management
- **Portfolio management**: Multi-asset portfolio construction and analysis
- **Transaction costs**: Realistic cost modeling for strategy evaluation
- **Performance metrics**: Sharpe ratio, drawdown, VaR, CVaR, and more

### 🔍 Explainability
- **SHAP analysis**: Feature importance and prediction explanations
- **Model diagnostics**: Residual analysis and model validation
- **Feature engineering**: Technical indicators, statistical features, regime detection

### 🎯 Scenario Analysis
- **What-if scenarios**: USD strength, interest rates, commodity shocks
- **Sensitivity analysis**: Factor impact and tornado charts
- **Stress testing**: Volatility and market regime changes

### 🌐 API Integration
- **REST API**: FastAPI endpoints for model predictions
- **Real-time predictions**: Get forecasts via HTTP requests
- **Model management**: Train and deploy models programmatically

## 📋 Supported Commodities

### Energy
- **WTI Crude Oil** (CL=F)
- **Brent Crude Oil** (BZ=F)
- **Natural Gas** (NG=F)

### Metals
- **Gold** (GC=F)
- **Silver** (SI=F)
- **Copper** (HG=F)

### Agriculture
- **Corn** (ZC=F)
- **Soybeans** (ZS=F)

### Macro Indicators
- **Dollar Index Proxy** (UUP)
- **S&P 500** (SPY)
- **10-Year Treasury Yield** (DGS10) - via FRED API

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- pip or conda

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd commodity-price-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional)
```bash
cp env.example .env
# Edit .env with your FRED API key
```

4. **Run the application**
```bash
streamlit run app/main.py
```

### Docker Installation

```bash
# Build the image
docker build -t commodity-predictor .

# Run the container
docker run -p 8501:8501 commodity-predictor
```

## 🚀 Usage

### 1. Overview Page
- **Market summary**: Key metrics and performance indicators
- **Data quality status**: Real-time data health monitoring
- **Quick actions**: Refresh data, train models, run backtests

### 2. Data Page
- **Interactive charts**: Price charts with technical indicators
- **Regime analysis**: Market regime detection and visualization
- **Feature engineering**: Technical indicators and statistical features
- **Data export**: Download data and analysis results

### 3. Models Page
- **Model training**: Train multiple models with different parameters
- **Performance comparison**: Side-by-side model evaluation
- **Walk-forward validation**: Time series cross-validation
- **Feature importance**: XGBoost feature importance analysis

### 4. Backtest Page
- **Strategy backtesting**: Test trading strategies with historical data
- **Portfolio backtesting**: Multi-asset portfolio analysis
- **Performance metrics**: Comprehensive risk and return analysis
- **Transaction costs**: Realistic cost modeling

### 5. Strategy Page
- **Strategy creation**: Define custom trading strategies
- **Risk management**: Stop-loss, take-profit, position sizing
- **Portfolio construction**: Multi-strategy portfolio management
- **Performance tracking**: Real-time strategy performance

### 6. Portfolio Page
- **Portfolio management**: Create and manage multi-asset portfolios
- **Risk analysis**: VaR, CVaR, correlation analysis
- **Performance attribution**: Strategy contribution analysis
- **Rebalancing**: Automated portfolio rebalancing

### 7. Explainability Page
- **SHAP analysis**: Model prediction explanations
- **Feature importance**: Understanding model decisions
- **Model diagnostics**: Residual analysis and validation
- **Partial dependence**: Feature effect analysis

### 8. Scenarios Page
- **Scenario analysis**: What-if analysis for different market conditions
- **Sensitivity analysis**: Factor impact and tornado charts
- **Stress testing**: Volatility and regime change scenarios
- **Custom scenarios**: User-defined market scenarios

### 9. Settings Page
- **Configuration**: App settings and preferences
- **API management**: REST API server controls
- **Data sources**: Configure data providers and caching
- **Performance tuning**: Optimization settings

## 🔧 Configuration

### Environment Variables
```bash
# FRED API Key for macroeconomic data
FRED_API_KEY=your_fred_api_key_here

# Random seed for reproducibility
RANDOM_SEED=42
```

### Streamlit Configuration
The app uses `.streamlit/config.toml` for configuration:
- **Theme**: Customizable color scheme
- **Layout**: Wide mode for better chart display
- **Caching**: Optimized data and model caching

## 📊 API Usage

### Start the API Server
```python
from services.api import start_api_server
start_api_server(host="127.0.0.1", port=8000)
```

### Make Predictions
```python
import requests

# Predict commodity price
response = requests.post("http://127.0.0.1:8000/predict", json={
    "symbol": "GC=F",
    "horizon": "1d",
    "model": "xgboost"
})

prediction = response.json()
print(f"Predicted return: {prediction['next_return']:.4f}")
print(f"Predicted price: ${prediction['next_price']:.2f}")
```

### Available Endpoints
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /predict` - Make predictions
- `GET /symbols` - Get available symbols
- `GET /horizons` - Get available horizons
- `GET /models` - Get available models
- `POST /train` - Train models

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test files:
```bash
pytest tests/test_metrics.py
pytest tests/test_backtest.py
pytest tests/test_features.py
pytest tests/test_arima.py
```

## 📈 Performance

### Data Processing
- **Caching**: Intelligent data caching with TTL
- **Parallel processing**: Multi-threaded model training
- **Memory optimization**: Efficient data structures and processing

### Model Training
- **Walk-forward validation**: Proper time series validation
- **Feature engineering**: Automated technical indicator calculation
- **Model selection**: Automatic hyperparameter tuning

### Backtesting
- **Transaction costs**: Realistic cost modeling
- **Risk metrics**: Comprehensive risk analysis
- **Performance tracking**: Real-time strategy performance

## 🔒 Security

### Data Protection
- **API keys**: Secure environment variable storage
- **Data validation**: Input sanitization and validation
- **Error handling**: Graceful error handling and logging

### Best Practices
- **No hardcoded secrets**: All sensitive data in environment variables
- **Input validation**: Comprehensive input validation
- **Error logging**: Detailed error logging and monitoring

## 🚀 Deployment

### Streamlit Cloud
1. **Fork the repository**
2. **Connect to Streamlit Cloud**
3. **Set environment variables**
4. **Deploy the application**

### Local Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app/main.py
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t commodity-predictor .
docker run -p 8501:8501 commodity-predictor
```

## 📚 Documentation

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Code Documentation
- **Docstrings**: Comprehensive function documentation
- **Type hints**: Full type annotation support
- **Comments**: Inline code comments and explanations

## 🤝 Contributing

### Development Setup
1. **Fork the repository**
2. **Create a feature branch**
3. **Install development dependencies**
4. **Make your changes**
5. **Run tests**
6. **Submit a pull request**

### Code Style
- **PEP 8**: Python code style guidelines
- **Type hints**: Full type annotation
- **Docstrings**: Google-style docstrings
- **Testing**: Comprehensive test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **yfinance**: Financial data provider
- **FRED**: Economic data from Federal Reserve
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library
- **XGBoost**: Gradient boosting framework
- **SHAP**: Model explainability library
- **Plotly**: Interactive visualization library

## 📞 Support

### Issues
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive user guide
- **Examples**: Sample code and use cases

### Community
- **Discussions**: Community discussions and Q&A
- **Wiki**: Additional documentation and guides
- **Contributing**: How to contribute to the project

## 🔮 Roadmap

### Upcoming Features
- **Real-time data**: Live data feeds and real-time predictions
- **Advanced models**: LSTM, Transformer, and ensemble methods
- **Risk management**: Advanced risk metrics and portfolio optimization
- **Mobile app**: Mobile application for commodity trading
- **Cloud deployment**: AWS, GCP, and Azure deployment options

### Version History
- **v1.0.0**: Initial release with basic functionality
- **v1.1.0**: Added SHAP analysis and explainability
- **v1.2.0**: Enhanced backtesting and portfolio management
- **v1.3.0**: Added scenario analysis and stress testing
- **v1.4.0**: REST API and deployment improvements

---

**Built with ❤️ for the commodity trading community**
