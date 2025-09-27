"""
FastAPI REST API for the Commodity Price Predictor.
"""

import os
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

from utils.constants import COMMODITIES, HORIZONS
from services.data_loader import DataLoader
from services.modeling import ModelTrainer
from services.features import FeatureEngineer

class PredictionRequest(BaseModel):
    symbol: str
    horizon: str = "1d"
    model: str = "xgboost"

class PredictionResponse(BaseModel):
    symbol: str
    horizon: str
    next_return: float
    next_price: float
    model: str
    timestamp: str
    confidence: Optional[float] = None

class APIHandler:
    """Handles the REST API for the Commodity Price Predictor."""
    
    def __init__(self, data_loader: DataLoader, model_trainer: ModelTrainer):
        self.data_loader = data_loader
        self.model_trainer = model_trainer
        self.feature_engineer = FeatureEngineer()
        self.trained_models = {}
        self.app = None
        self.server_thread = None
        self.server_running = False
        
        if API_AVAILABLE:
            self._setup_api()
    
    def _setup_api(self):
        """Setup FastAPI application."""
        if not API_AVAILABLE:
            return
        
        self.app = FastAPI(
            title="Commodity Price Predictor API",
            description="REST API for commodity price predictions",
            version="1.0.0"
        )
        
        # Add routes
        self.app.add_api_route("/", self.root, methods=["GET"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/predict", self.predict, methods=["POST"])
        self.app.add_api_route("/symbols", self.get_symbols, methods=["GET"])
        self.app.add_api_route("/horizons", self.get_horizons, methods=["GET"])
        self.app.add_api_route("/models", self.get_models, methods=["GET"])
        self.app.add_api_route("/train", self.train_model, methods=["POST"])
    
    def start_server(self, host: str = "127.0.0.1", port: int = 8000):
        """Start the API server in a separate thread."""
        if not API_AVAILABLE:
            raise RuntimeError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        if self.server_running:
            return {"message": "Server already running"}
        
        def run_server():
            uvicorn.run(self.app, host=host, port=port, log_level="info")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.server_running = True
        
        return {"message": f"Server started on http://{host}:{port}"}
    
    def stop_server(self):
        """Stop the API server."""
        self.server_running = False
        return {"message": "Server stopped"}
    
    async def root(self):
        """Root endpoint."""
        return {
            "message": "Commodity Price Predictor API",
            "version": "1.0.0",
            "endpoints": [
                "/health",
                "/predict",
                "/symbols",
                "/horizons",
                "/models",
                "/train"
            ]
        }
    
    async def health(self):
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": len(self.trained_models)
        }
    
    async def predict(self, request: PredictionRequest):
        """Predict endpoint."""
        try:
            # Validate symbol
            if request.symbol not in COMMODITIES:
                raise HTTPException(status_code=400, detail=f"Invalid symbol: {request.symbol}")
            
            # Validate horizon
            if request.horizon not in HORIZONS:
                raise HTTPException(status_code=400, detail=f"Invalid horizon: {request.horizon}")
            
            # Get data
            data = self.data_loader.load_commodity_data([request.symbol])
            if request.symbol not in data or data[request.symbol].empty:
                raise HTTPException(status_code=404, detail=f"No data available for {request.symbol}")
            
            df = data[request.symbol]
            
            # Create features
            df = self.feature_engineer.create_features(df, request.symbol)
            df = self.feature_engineer.create_targets(df, [request.horizon])
            
            # Get prediction (simplified - in practice would use trained models)
            if f'target_{request.horizon}' in df.columns:
                next_return = df[f'target_{request.horizon}'].iloc[-1]
            else:
                # Fallback to simple momentum
                horizon_days = HORIZONS[request.horizon]['days']
                next_return = df['Close'].pct_change(horizon_days).iloc[-1]
            
            # Calculate next price
            current_price = df['Close'].iloc[-1]
            next_price = current_price * (1 + next_return)
            
            # Calculate confidence (simplified)
            confidence = min(0.95, max(0.05, 1 - abs(next_return) * 10))
            
            response = PredictionResponse(
                symbol=request.symbol,
                horizon=request.horizon,
                next_return=next_return,
                next_price=next_price,
                model=request.model,
                timestamp=datetime.now().isoformat(),
                confidence=confidence
            )
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_symbols(self):
        """Get available symbols."""
        return {
            "symbols": list(COMMODITIES.keys()),
            "commodities": COMMODITIES
        }
    
    async def get_horizons(self):
        """Get available horizons."""
        return {
            "horizons": list(HORIZONS.keys()),
            "horizon_details": HORIZONS
        }
    
    async def get_models(self):
        """Get available models."""
        return {
            "models": ["naive_last", "naive_seasonal", "arima", "xgboost", "prophet"],
            "trained_models": list(self.trained_models.keys())
        }
    
    async def train_model(self, request: Dict[str, Any]):
        """Train a model for a symbol."""
        try:
            symbol = request.get('symbol')
            horizon = request.get('horizon', '1d')
            model_name = request.get('model', 'xgboost')
            
            if not symbol or symbol not in COMMODITIES:
                raise HTTPException(status_code=400, detail="Invalid symbol")
            
            # Get data
            data = self.data_loader.load_commodity_data([symbol])
            if symbol not in data or data[symbol].empty:
                raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
            
            df = data[symbol]
            
            # Create features
            df = self.feature_engineer.create_features(df, symbol)
            df = self.feature_engineer.create_targets(df, [horizon])
            
            # Train model
            results = self.model_trainer.train_models(df, symbol, [horizon])
            
            if horizon in results and 'error' not in results[horizon]:
                # Store trained model
                model_key = f"{symbol}_{horizon}_{model_name}"
                self.trained_models[model_key] = results[horizon]
                
                return {
                    "message": f"Model trained successfully for {symbol} {horizon}",
                    "model_key": model_key,
                    "metrics": results[horizon].get('metrics', {})
                }
            else:
                raise HTTPException(status_code=500, detail="Model training failed")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status information."""
        return {
            "api_available": API_AVAILABLE,
            "server_running": self.server_running,
            "trained_models": len(self.trained_models),
            "model_keys": list(self.trained_models.keys())
        }
    
    def get_api_docs_url(self, host: str = "127.0.0.1", port: int = 8000) -> str:
        """Get API documentation URL."""
        return f"http://{host}:{port}/docs"
    
    def get_api_redoc_url(self, host: str = "127.0.0.1", port: int = 8000) -> str:
        """Get API ReDoc URL."""
        return f"http://{host}:{port}/redoc"

# Global API handler instance
api_handler = None

def get_api_handler() -> Optional[APIHandler]:
    """Get the global API handler instance."""
    return api_handler

def initialize_api(data_loader: DataLoader, model_trainer: ModelTrainer) -> APIHandler:
    """Initialize the API handler."""
    global api_handler
    api_handler = APIHandler(data_loader, model_trainer)
    return api_handler

def start_api_server(host: str = "127.0.0.1", port: int = 8000) -> Dict[str, Any]:
    """Start the API server."""
    if api_handler is None:
        return {"error": "API handler not initialized"}
    
    return api_handler.start_server(host, port)

def stop_api_server() -> Dict[str, Any]:
    """Stop the API server."""
    if api_handler is None:
        return {"error": "API handler not initialized"}
    
    return api_handler.stop_server()

def get_api_status() -> Dict[str, Any]:
    """Get API status."""
    if api_handler is None:
        return {"error": "API handler not initialized"}
    
    return api_handler.get_api_status()
