#!/usr/bin/env python3
"""
Simple test to check if the app can start
"""

import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent / "app"
sys.path.append(str(app_dir))

try:
    from app.utils.constants import COMMODITIES
    print("✅ Constants imported successfully")
    print(f"Found {len(COMMODITIES)} commodities")
except Exception as e:
    print(f"❌ Constants import error: {e}")

try:
    from app.services.data_loader import DataLoader
    print("✅ DataLoader imported successfully")
except Exception as e:
    print(f"❌ DataLoader import error: {e}")

print("Test completed!")
