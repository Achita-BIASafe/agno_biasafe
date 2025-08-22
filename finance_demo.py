"""
BiasafeAI Finance Tools - Standalone Demo

This demonstrates the new team-based architecture where each financial 
analysis function is available as an individual tool.

Run with: python finance_demo.py
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from tools.finance_client import create_client
from tools.finance_tools import list_tools, get_tool_info

def main():
    """Run the finance tools demo."""
    
    print("=" * 80)
    print("🏦 BIASAFEAI FINANCE TOOLS - STANDALONE DEMO")
    print("=" * 80)
    print()
    
    # Create client
    print("🔧 Initializing Finance Client...")
    client = create_client()
    
    # Show status
    print("\n📊 Client Status:")
    status = client.status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Show available tools
    print("\n🛠️ Available Financial Tools:")
    print("=" * 50)
    
    tools = list_tools()
    tool_info = get_tool_info()
    
    for category, tool_list in tool_info["categories"].items():
        print(f"\n📊 {category}:")
        for tool in tool_list:
            print(f"      • {tool}")
    
    print(f"\nTotal: {len(tools)} specialized financial analysis tools")
    
    # Interactive demo
    print("\n" + "=" * 80)
    print("🎯 INTERACTIVE DEMO")
    print("=" * 80)
    
    while True:
        print("\nChoose an option:")
        print("1. Direct tool usage examples")
        print("2. Test portfolio optimization")
        print("3. Test risk analysis")
        print("4. Test market data analysis")
        print("5. Test backtesting")
        if client.conversational_mode:
            print("6. Conversational chat interface")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == "0":
            print("\n👋 Thanks for using BiasafeAI Finance Tools!")
            break
            
        elif choice == "1":
            show_usage_examples()
            
        elif choice == "2":
            demo_portfolio_optimization(client)
            
        elif choice == "3":
            demo_risk_analysis(client)
            
        elif choice == "4":
            demo_market_data(client)
            
        elif choice == "5":
            demo_backtesting(client)
            
        elif choice == "6" and client.conversational_mode:
            demo_chat(client)
            
        else:
            print("❌ Invalid choice. Please try again.")

def show_usage_examples():
    """Show code usage examples."""
    print("\n" + "=" * 60)
    print("💻 DIRECT TOOL USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        ("Portfolio Optimization", "client.portfolio.optimize_portfolio(assets='AAPL,MSFT,GOOGL')"),
        ("Risk Metrics", "client.risk.calculate_risk_metrics(assets='AAPL,MSFT,GOOGL')"),
        ("Market Analysis", "client.data.analyze_market_data(assets='AAPL,MSFT,GOOGL')"),
        ("Backtesting", "client.backtest.backtest_strategy(assets='AAPL,MSFT,GOOGL')"),
        ("Performance Metrics", "client.performance.get_performance_metrics(assets='AAPL,MSFT,GOOGL')"),
        ("Correlation Analysis", "client.correlation.analyze_correlations(assets='AAPL,MSFT,GOOGL,TSLA')"),
    ]
    
    for name, code in examples:
        print(f"\n{name}:")
        print(f"  {code}")

def demo_portfolio_optimization(client):
    """Demo portfolio optimization."""
    print("\n" + "=" * 60)
    print("📈 PORTFOLIO OPTIMIZATION DEMO")
    print("=" * 60)
    
    assets = input("Enter assets (comma-separated, e.g. AAPL,MSFT,GOOGL): ").strip()
    if not assets:
        assets = "AAPL,MSFT,GOOGL"
    
    print(f"\n🔍 Optimizing portfolio for: {assets}")
    
    try:
        result = client.portfolio.optimize_portfolio(assets=assets)
        print(f"✅ Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_risk_analysis(client):
    """Demo risk analysis."""
    print("\n" + "=" * 60)
    print("⚠️ RISK ANALYSIS DEMO")
    print("=" * 60)
    
    assets = input("Enter assets (comma-separated, e.g. AAPL,MSFT,GOOGL): ").strip()
    if not assets:
        assets = "AAPL,MSFT,GOOGL"
    
    print(f"\n🔍 Analyzing risk for: {assets}")
    
    try:
        result = client.risk.calculate_risk_metrics(assets=assets)
        print(f"✅ Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_market_data(client):
    """Demo market data analysis."""
    print("\n" + "=" * 60)
    print("📊 MARKET DATA ANALYSIS DEMO")
    print("=" * 60)
    
    assets = input("Enter assets (comma-separated, e.g. AAPL,MSFT,GOOGL): ").strip()
    if not assets:
        assets = "AAPL,MSFT,GOOGL"
    
    print(f"\n🔍 Analyzing market data for: {assets}")
    
    try:
        result = client.data.analyze_market_data(assets=assets)
        print(f"✅ Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_backtesting(client):
    """Demo backtesting."""
    print("\n" + "=" * 60)
    print("🔄 BACKTESTING DEMO")
    print("=" * 60)
    
    assets = input("Enter assets (comma-separated, e.g. AAPL,MSFT,GOOGL): ").strip()
    if not assets:
        assets = "AAPL,MSFT,GOOGL"
    
    print(f"\n🔍 Backtesting strategy for: {assets}")
    
    try:
        result = client.backtest.backtest_strategy(assets=assets)
        print(f"✅ Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_chat(client):
    """Demo conversational interface."""
    print("\n" + "=" * 60)
    print("💬 CONVERSATIONAL CHAT DEMO")
    print("=" * 60)
    print("Type 'exit' to return to main menu")
    
    while True:
        user_input = input("\n🗣️ You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'back']:
            break
            
        if not user_input:
            continue
            
        try:
            print("🤖 Finance Team: Processing your request...")
            response = client.chat(user_input)
            print(f"🤖 Finance Team: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
