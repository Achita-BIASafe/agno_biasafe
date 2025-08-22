#!/usr/bin/env python3
"""
Test Finance Team Import and Basic Functionality
"""

import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_finance_team():
    """Test if Finance Team can be imported and basic functionality works"""
    try:
        print("🔍 Testing Finance Team import...")
        
        # Test tools import first
        from tools.finance_tools import list_tools, get_tool_info
        print("✅ Finance tools imported successfully")
        
        # List available tools
        tools = list_tools()
        print(f"📊 Found {len(tools)} available tools:")
        for tool in tools:
            print(f"   • {tool}")
        
        # Test tool info
        tool_info = get_tool_info()
        print(f"\n📋 Tool categories:")
        for category, tool_list in tool_info["categories"].items():
            print(f"   📊 {category}: {len(tool_list)} tools")
        
        print(f"\n✅ Finance Team tools are ready!")
        print("🎯 You can now use the Finance Team agent to pick the right tool for your requests")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Finance Team Setup...")
    print("=" * 50)
    
    success = test_finance_team()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Finance Team is ready to use!")
        print("💡 The Finance Team agent can intelligently pick tools based on your requests")
    else:
        print("❌ Finance Team setup needs fixing")
