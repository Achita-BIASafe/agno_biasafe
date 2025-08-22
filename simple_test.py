#!/usr/bin/env python3
"""
Simple Finance Team Test
"""

print("Starting test...")

try:
    import sys
    import os
    print("✅ Basic imports successful")
    
    # Add current directory 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    print("✅ Path setup complete")
    
    # Test finance_tools import step by step
    print("🔍 Testing finance_tools import...")
    from tools import finance_tools
    print("✅ finance_tools module imported")
    
    # Test list_tools function
    tools = finance_tools.list_tools()
    print(f"✅ Found {len(tools)} tools: {tools}")
    
    print("🎯 SUCCESS! Finance Team tools are working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
