"""
Financial Analysis Client

This module provides a unified interface for accessing financial analysis tools
either directly or through conversational interaction.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the finance tools and team
from finance_tools import finance_tools, get_tool_info, list_tools
from finance_team import FinanceTeam

# Load environment variables
load_dotenv()


class FinanceClient:
    """
    Unified client for accessing financial analysis capabilities.
    
    Provides both direct tool access and conversational interaction
    through the finance team agent.
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the finance client.
        
        Args:
            openai_api_key: OpenAI API key for conversational features
        """
        self.tools = finance_tools
        
        # Get API key from parameter or environment
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Create specialized tool interfaces
        self.portfolio = PortfolioTools()
        self.risk = RiskTools()
        self.data = DataTools()
        self.backtest = BacktestTools()
        
        # Create conversational team agent if API key is available
        if self.openai_api_key:
            try:
                self.team = FinanceTeam(
                    agent_id="finance-team",
                    name="Finance Team",
                    description="Financial analysis team with portfolio optimization, risk analysis, data analysis, and backtesting capabilities."
                )
                self.conversational_mode = True
            except Exception as e:
                print(f"Warning: Could not initialize conversational team: {e}")
                self.team = None
                self.conversational_mode = False
        else:
            print("Warning: No OpenAI API key provided. Conversational features will be unavailable.")
            self.team = None
            self.conversational_mode = False
    
    def chat(self, message: str) -> str:
        """
        Chat with the finance team using natural language.
        
        Args:
            message: Natural language request for financial analysis
            
        Returns:
            Response from the finance team
        """
        if not self.conversational_mode:
            return "Conversational mode is not available. Please provide an OpenAI API key to enable this feature."
        
        try:
            return self.team.run(message)
        except Exception as e:
            return f"Error in conversational mode: {str(e)}"
    
    def get_tool(self, name: str):
        """Get a specific tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all available tools."""
        return list_tools()
    
    def get_tool_info(self, tool_name: str = None) -> Dict[str, Any]:
        """Get information about available tools."""
        return get_tool_info(tool_name)
    
    def status(self) -> Dict[str, Any]:
        """Get client status information."""
        return {
            "tools_available": len(self.tools),
            "conversational_mode": self.conversational_mode,
            "openai_api_key_configured": bool(self.openai_api_key),
            "categories": {
                "Portfolio Optimization": len([t for t in self.tools.keys() if any(kw in t for kw in ["optimize", "frontier"])]),
                "Risk Analysis": len([t for t in self.tools.keys() if any(kw in t for kw in ["risk", "var", "stress"])]),
                "Data Analysis": len([t for t in self.tools.keys() if any(kw in t for kw in ["data", "correlation", "quality"])]),
                "Backtesting": len([t for t in self.tools.keys() if any(kw in t for kw in ["backtest", "strategy", "drawdown"])])
            }
        }


class PortfolioTools:
    """Interface for portfolio optimization tools."""
    
    def optimize_portfolio(self, **kwargs) -> Dict[str, Any]:
        """Optimize a portfolio based on modern portfolio theory."""
        return finance_tools["optimize_portfolio"](**kwargs)
    
    def create_efficient_frontier(self, **kwargs) -> Dict[str, Any]:
        """Generate efficient frontier for a set of assets."""
        return finance_tools["create_efficient_frontier"](**kwargs)


class RiskTools:
    """Interface for risk analysis tools."""
    
    def calculate_risk_metrics(self, **kwargs) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for a portfolio."""
        return finance_tools["calculate_risk_metrics"](**kwargs)
    
    def stress_test_portfolio(self, **kwargs) -> Dict[str, Any]:
        """Perform stress testing on a portfolio."""
        return finance_tools["stress_test_portfolio"](**kwargs)
    
    def calculate_var_cvar(self, **kwargs) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)."""
        return finance_tools["calculate_var_cvar"](**kwargs)


class DataTools:
    """Interface for data analysis tools."""
    
    def analyze_market_data(self, **kwargs) -> Dict[str, Any]:
        """Analyze market data for specific assets."""
        return finance_tools["analyze_market_data"](**kwargs)
    
    def calculate_correlation_matrix(self, **kwargs) -> Dict[str, Any]:
        """Calculate correlation matrix between assets."""
        return finance_tools["calculate_correlation_matrix"](**kwargs)
    
    def check_data_quality(self, **kwargs) -> Dict[str, Any]:
        """Check data quality for specified assets."""
        return finance_tools["check_data_quality"](**kwargs)


class BacktestTools:
    """Interface for backtesting tools."""
    
    def backtest_strategy(self, **kwargs) -> Dict[str, Any]:
        """Backtest an investment strategy on historical data."""
        return finance_tools["backtest_strategy"](**kwargs)
    
    def compare_strategies(self, **kwargs) -> Dict[str, Any]:
        """Compare multiple investment strategies side by side."""
        return finance_tools["compare_strategies"](**kwargs)
    
    def analyze_drawdowns(self, **kwargs) -> Dict[str, Any]:
        """Analyze portfolio drawdowns over time."""
        return finance_tools["analyze_drawdowns"](**kwargs)


def create_client(openai_api_key: str = None) -> FinanceClient:
    """
    Create and return a finance client instance.
    
    Args:
        openai_api_key: OpenAI API key for conversational features
        
    Returns:
        Configured FinanceClient instance
    """
    return FinanceClient(openai_api_key)


def demo_usage():
    """Demonstrate various ways to use the finance client."""
    print("=" * 60)
    print("FINANCIAL ANALYSIS CLIENT DEMO")
    print("=" * 60)
    
    # Create client
    client = create_client()
    
    # Show status
    print("\n1. CLIENT STATUS:")
    status = client.status()
    print(f"   Tools available: {status['tools_available']}")
    print(f"   Conversational mode: {status['conversational_mode']}")
    print(f"   Tool categories: {status['categories']}")
    
    # List available tools
    print("\n2. AVAILABLE TOOLS:")
    tools = client.list_tools()
    for i, tool in enumerate(tools, 1):
        print(f"   {i:2d}. {tool}")
    
    # Example direct tool usage
    print("\n3. DIRECT TOOL USAGE EXAMPLES:")
    print("   # Portfolio optimization")
    print('   client.portfolio.optimize_portfolio(assets="AAPL,MSFT,GOOGL")')
    print()
    print("   # Risk analysis")
    print('   client.risk.calculate_risk_metrics(assets="AAPL,MSFT,GOOGL")')
    print()
    print("   # Data analysis")
    print('   client.data.analyze_market_data(assets="AAPL,MSFT,GOOGL")')
    print()
    print("   # Backtesting")
    print('   client.backtest.backtest_strategy(assets="AAPL,MSFT,GOOGL", strategy="equal_weight")')
    
    # Example conversational usage
    print("\n4. CONVERSATIONAL USAGE EXAMPLES:")
    if client.conversational_mode:
        print("   # Natural language requests")
        print('   client.chat("Optimize a portfolio with AAPL, MSFT, and GOOGL")')
        print('   client.chat("What\'s the risk of holding equal weights in tech stocks?")')
        print('   client.chat("Backtest a momentum strategy over the past year")')
    else:
        print("   Conversational mode requires OpenAI API key configuration")
        print("   Set OPENAI_API_KEY environment variable or pass to create_client()")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run demo if executed directly
    demo_usage()
    
    # Interactive mode
    print("\nStarting interactive mode...")
    client = create_client()
    
    if client.conversational_mode:
        print("You can now chat with the finance team!")
        print("Type 'quit' to exit, 'tools' to list available tools, 'status' for client info")
        print()
        
        while True:
            try:
                user_input = input("💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'tools':
                    tools = client.list_tools()
                    print(f"Available tools ({len(tools)}):")
                    for tool in tools:
                        print(f"  • {tool}")
                elif user_input.lower() == 'status':
                    status = client.status()
                    for key, value in status.items():
                        print(f"  {key}: {value}")
                elif user_input:
                    response = client.chat(user_input)
                    print(f"🤖 Finance Team: {response}")
                    print()
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("Interactive conversational mode requires OpenAI API key.")
        print("You can still use direct tool access:")
        print("  client.portfolio.optimize_portfolio(assets='AAPL,MSFT,GOOGL')")
        print("  client.risk.calculate_risk_metrics(assets='AAPL,MSFT,GOOGL')")
        print("  etc.")
