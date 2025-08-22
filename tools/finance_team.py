"""
Finance Team Agent - Simplified Version with Tool Selection

This provides the Finance Team agent that can intelligently select and route
to appropriate financial analysis tools without circular import issues.
"""

from agno.agent import Agent
from agno.run.response import RunResponse
from typing import Iterator, List, Dict, Any, Optional
import re


class FinanceTeam(Agent):
    """
    A Finance Team agent that intelligently selects which financial tool to use
    based on user requests. This agent acts as a router and can pick the best
    tool for each financial analysis task.
    """
    
    name = "Finance Team"
    role = "Expert Financial Analysis Team - Tool Selector"
    
    instructions = [
        "You are a financial analysis expert that helps users by selecting the right tool for their needs.",
        "You can analyze user requests and recommend the best financial analysis approach.",
        "Available tool categories:",
        "- Portfolio Optimization: optimize_portfolio, calculate_efficient_frontier, calculate_sharpe_ratio",
        "- Risk Analysis: calculate_risk_metrics, stress_test_portfolio", 
        "- Market Analysis: analyze_market_data, analyze_correlation_matrix",
        "- Strategy Testing: backtest_strategy",
        "- Reporting: generate_financial_report, validate_portfolio_inputs",
        "Extract asset symbols and parameters from natural language.",
        "Provide clear, actionable recommendations for which tools to use.",
        "Always explain your reasoning for tool selection."
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Tool information without importing the actual tools
        self.available_tools = {
            "Portfolio Optimization": [
                "optimize_portfolio",
                "calculate_efficient_frontier", 
                "calculate_sharpe_ratio"
            ],
            "Risk Analysis": [
                "calculate_risk_metrics",
                "stress_test_portfolio"
            ],
            "Market Analysis": [
                "analyze_market_data",
                "analyze_correlation_matrix"
            ],
            "Strategy Testing": [
                "backtest_strategy"
            ],
            "Reporting": [
                "generate_financial_report",
                "validate_portfolio_inputs"
            ]
        }
        
        self.total_tools = sum(len(tools) for tools in self.available_tools.values())
        
        self.description = f"""
        Finance Team - Intelligent Tool Selector
        
        I help you choose the right financial analysis tool for your needs from {self.total_tools} available tools:
        
        📊 Portfolio Optimization (3 tools):
        - Create optimal portfolios with different risk measures
        - Generate efficient frontiers 
        - Calculate risk-adjusted returns
        
        🎯 Risk Analysis (2 tools):
        - Comprehensive risk metrics (VaR, CVaR, volatility)
        - Stress testing under different scenarios
        
        📈 Market Analysis (2 tools):
        - Market data analysis and trends
        - Asset correlation analysis
        
        🔄 Strategy Testing (1 tool):
        - Historical strategy backtesting
        
        📋 Reporting (2 tools):
        - Generate comprehensive reports
        - Validate portfolio inputs
        
        Just tell me what you want to analyze and I'll recommend the best tool!
        """
    
    def run(self, message: str, **kwargs) -> Iterator[RunResponse]:
        """Process the user request and recommend appropriate tools."""
        try:
            # Initial progress message
            yield RunResponse(content="🔄 **Analyzing your request...**\n\nDetermining the best financial analysis approach...")
            
            response_content = self.determine_best_tools(message)
            
            # Final response
            yield RunResponse(content=str(response_content))
            
        except Exception as e:
            error_message = f"❌ **Error Processing Request**\n\nI encountered an error: {str(e)}. Please try rephrasing your question."
            yield RunResponse(content=error_message)
    
    def determine_best_tools(self, message: str) -> str:
        """Analyze the user's request and recommend the best tools."""
        message_lower = message.lower()
        
        # Extract assets from the message
        assets = self.extract_assets(message)
        
        recommended_tools = []
        reasoning = []
        
        # Portfolio Optimization Analysis
        if any(kw in message_lower for kw in ["optimize", "portfolio allocation", "weights", "allocation", "optimal", "efficient frontier"]):
            if "efficient frontier" in message_lower:
                recommended_tools.append("calculate_efficient_frontier")
                reasoning.append("📊 **Efficient Frontier**: Generate risk/return optimization curve")
            elif "sharpe" in message_lower:
                recommended_tools.append("calculate_sharpe_ratio")
                reasoning.append("📊 **Sharpe Ratio**: Calculate risk-adjusted returns")
            else:
                recommended_tools.append("optimize_portfolio")
                reasoning.append("📊 **Portfolio Optimization**: Create optimal asset allocation")
        
        # Risk Analysis
        if any(kw in message_lower for kw in ["risk", "var", "volatility", "drawdown", "cvar"]):
            if "stress" in message_lower:
                recommended_tools.append("stress_test_portfolio")
                reasoning.append("🎯 **Stress Testing**: Test portfolio under adverse scenarios")
            else:
                recommended_tools.append("calculate_risk_metrics")
                reasoning.append("🎯 **Risk Metrics**: Comprehensive risk analysis (VaR, CVaR, volatility)")
        
        # Market Data Analysis
        if any(kw in message_lower for kw in ["analyze data", "market data", "trends", "correlation"]):
            if "correlation" in message_lower:
                recommended_tools.append("analyze_correlation_matrix")
                reasoning.append("📈 **Correlation Analysis**: Asset correlation relationships")
            else:
                recommended_tools.append("analyze_market_data")
                reasoning.append("📈 **Market Analysis**: Market trends and statistics")
        
        # Backtesting
        if any(kw in message_lower for kw in ["backtest", "strategy", "historical performance", "compare strategies"]):
            recommended_tools.append("backtest_strategy")
            reasoning.append("🔄 **Strategy Backtesting**: Historical performance analysis")
        
        # Reporting
        if any(kw in message_lower for kw in ["report", "summary", "comprehensive"]):
            recommended_tools.append("generate_financial_report")
            reasoning.append("📋 **Financial Report**: Comprehensive analysis summary")
        
        # Build response
        if recommended_tools:
            response = f"🎯 **Recommended Financial Analysis Tools**\n\n"
            response += f"**Assets detected:** {', '.join(assets) if assets else 'Default portfolio (AAPL, MSFT, GOOGL, AMZN)'}\n\n"
            response += f"**Your request analysis:** {message}\n\n"
            response += f"**Recommended tools:**\n"
            
            for tool, reason in zip(recommended_tools, reasoning):
                response += f"\n{reason}\n"
                response += f"   **Tool:** `{tool}`\n"
                response += f"   **Assets:** `{','.join(assets) if assets else 'AAPL,MSFT,GOOGL,AMZN'}`\n"
            
            response += f"\n**Next Steps:**\n"
            response += f"1. The recommended tools can be called directly by LLMs\n"
            response += f"2. Each tool will analyze your specified assets\n"
            response += f"3. Results will provide actionable financial insights\n"
            
            return response
        else:
            return self.provide_general_guidance(assets)
    
    def extract_assets(self, message: str) -> List[str]:
        """Extract stock symbols from the message."""
        # Match common stock patterns
        pattern = r'\b[A-Z]{1,5}(?:-[A-Z]+)?\b'
        symbols = re.findall(pattern, message.upper())
        
        # Filter out common words
        common_words = {'AND', 'OR', 'THE', 'FOR', 'WITH', 'FROM', 'TO', 'IN', 'ON', 'AT', 'BY', 'OF', 'VAR', 'CVAR'}
        symbols = [s for s in symbols if s not in common_words]
        
        return symbols[:10] if symbols else []
    
    def provide_general_guidance(self, assets: List[str]) -> str:
        """Provide general guidance when specific intent isn't clear."""
        response = f"🏦 **Finance Team - Tool Selection Guide**\n\n"
        
        if assets:
            response += f"**Assets mentioned:** {', '.join(assets)}\n\n"
        
        response += f"I can help you choose from {self.total_tools} financial analysis tools:\n\n"
        
        for category, tools in self.available_tools.items():
            response += f"**{category}** ({len(tools)} tools):\n"
            for tool in tools:
                response += f"   • `{tool}`\n"
            response += "\n"
        
        response += f"**Example requests:**\n"
        response += f"• 'Optimize a portfolio with AAPL, MSFT, GOOGL'\n"
        response += f"• 'Calculate risk metrics for tech stocks'\n"
        response += f"• 'Analyze correlation between TSLA and AMZN'\n"
        response += f"• 'Backtest a momentum strategy'\n"
        response += f"• 'Stress test my portfolio under recession scenario'\n\n"
        
        response += f"Just tell me what you'd like to analyze!"
        
        return response
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Get information about all available tools."""
        return {
            "categories": self.available_tools,
            "total_tools": self.total_tools,
            "tools_available": True,
            "description": "Finance Team with intelligent tool selection"
        }
