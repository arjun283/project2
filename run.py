#!/usr/bin/env python3
"""
Commodity Price Predictor - Run Script
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        print("✅ Core dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_data_directory():
    """Check if data directory exists and create if needed."""
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
        print("📁 Created data directory")
    else:
        print("📁 Data directory exists")

def check_artifacts_directory():
    """Check if artifacts directory exists and create if needed."""
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        artifacts_dir.mkdir()
        print("📁 Created artifacts directory")
    else:
        print("📁 Artifacts directory exists")

def run_app(port=8502):
    """Run the Streamlit application."""
    try:
        print(f"🚀 Starting Commodity Price Predictor on port {port}...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app/main.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main function."""
    print("📈 Commodity Price Predictor")
    print("=" * 40)
    
    # Get port from command line arguments or use default
    port = 8502
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            print(f"Using port: {port}")
        except ValueError:
            print(f"Invalid port '{sys.argv[1]}', using default port: {port}")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check directories
    check_data_directory()
    check_artifacts_directory()
    
    # Run the application
    run_app(port)

if __name__ == "__main__":
    main()
