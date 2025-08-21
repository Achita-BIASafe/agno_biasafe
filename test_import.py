#!/usr/bin/env python3
"""Test script to check imports"""

try:
    from risk_analyzer import RiskAnalyzer
    print("SUCCESS: RiskAnalyzer imported successfully")
    
    # Try to instantiate
    ra = RiskAnalyzer()
    print("SUCCESS: RiskAnalyzer instantiated successfully")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
