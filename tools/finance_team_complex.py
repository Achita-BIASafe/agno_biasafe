"""
Financial Analysis Team for Agno Framework

This team provides specialized financial analysis tools that can be called individually
or accessed through a conversational interface.
"""

from agno.agent import Agent
from agno.run.response import RunResponse
from typing import Iterator
import re
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import existing workflow components
from workflows.portfolio_optimizer import PortfolioOptimizer
from workflows.risk_analyzer import RiskAnalyzer
from workflows.data_analyzer import DataAnalyzer
from workflows.portfolio_backtester import PortfolioBacktester

# Import the tools for direct access
try:
    # Use a local import to avoid circular dependency
    import importlib
    finance_tools_module = importlib.import_module('.finance_tools', package='tools')
    
    optimize_portfolio = getattr(finance_tools_module, 'optimize_portfolio', None)
    calculate_risk_metrics = getattr(finance_tools_module, 'calculate_risk_metrics', None)
    backtest_strategy = getattr(finance_tools_module, 'backtest_strategy', None)
    stress_test_portfolio = getattr(finance_tools_module, 'stress_test_portfolio', None)
    analyze_market_data = getattr(finance_tools_module, 'analyze_market_data', None)
    calculate_efficient_frontier = getattr(finance_tools_module, 'calculate_efficient_frontier', None)
    analyze_correlation_matrix = getattr(finance_tools_module, 'analyze_correlation_matrix', None)
    calculate_sharpe_ratio = getattr(finance_tools_module, 'calculate_sharpe_ratio', None)
    generate_financial_report = getattr(finance_tools_module, 'generate_financial_report', None)
    validate_portfolio_inputs = getattr(finance_tools_module, 'validate_portfolio_inputs', None)
    get_tool_info = getattr(finance_tools_module, 'get_tool_info', None)
    list_tools = getattr(finance_tools_module, 'list_tools', None)
    
    TOOLS_AVAILABLE = True
except Exception:
    # If there's any import issue, we'll handle it differently
    TOOLS_AVAILABLE = False


class FinanceTeam(Agent):
    """
    A team of financial analysis tools that provides portfolio optimization,
    risk analysis, data analysis, and backtesting capabilities.
    
    Each tool corresponds to a specific API function that can be called directly.
    """
    
    name = "Finance Team"
    role = "Expert Financial Analysis Team"
    
    instructions = [
        "You are a team of financial analysis experts providing specialized tools:",
        "- Portfolio optimization with modern portfolio theory",
        "- Risk analysis including VaR, CVaR, and stress testing", 
        "- Market data analysis and correlation studies",
        "- Strategy backtesting and performance comparison",
        "Analyze user requests to determine which tool(s) to use.",
        "Extract asset symbols and parameters from natural language.",
        "Provide clear, actionable financial insights.",
        "Always explain your methodology and assumptions."
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize workflow components
        self.portfolio_optimizer = PortfolioOptimizer()
        self.risk_analyzer = RiskAnalyzer()
        self.data_analyzer = DataAnalyzer()
        self.portfolio_backtester = PortfolioBacktester()
        
        # Set up tools access
        if TOOLS_AVAILABLE:
            # Direct access to individual tools
            self.tools = {
                'optimize_portfolio': optimize_portfolio,
                'calculate_risk_metrics': calculate_risk_metrics,
                'backtest_strategy': backtest_strategy,
                'stress_test_portfolio': stress_test_portfolio,
                'analyze_market_data': analyze_market_data,
                'calculate_efficient_frontier': calculate_efficient_frontier,
                'analyze_correlation_matrix': analyze_correlation_matrix,
                'calculate_sharpe_ratio': calculate_sharpe_ratio,
                'generate_financial_report': generate_financial_report,
                'validate_portfolio_inputs': validate_portfolio_inputs
            }
            
            # Get tool information for description
            try:
                tool_info = get_tool_info()
                available_tools = list_tools()
                self.available_tools_count = len(available_tools)
                self.tool_categories = tool_info.get("categories", {})
            except:
                self.available_tools_count = len(self.tools)
                self.tool_categories = {}
        else:
            self.tools = {}
            self.available_tools_count = 0
            self.tool_categories = {}
        
        self.description = f"""
        FinanceTeam provides comprehensive financial analysis with {self.available_tools_count} specialized tools:
        
        📊 Portfolio Optimization:
        - optimize_portfolio: Create optimal asset allocations
        - calculate_efficient_frontier: Generate risk/return frontier
        - calculate_sharpe_ratio: Risk-adjusted return metrics
        
        🎯 Risk Analysis:
        - calculate_risk_metrics: VaR, CVaR, volatility analysis
        - stress_test_portfolio: Scenario and stress testing
        
        📈 Market Analysis:
        - analyze_market_data: Market trends and statistics
        - analyze_correlation_matrix: Asset correlation analysis
        
        🔄 Strategy Testing:
        - backtest_strategy: Historical performance testing
        
        📋 Reporting:
        - generate_financial_report: Comprehensive analysis reports
        - validate_portfolio_inputs: Input validation and checks
        
        Each tool performs a specific financial analysis function and can be called directly by LLMs.
        """
        
    def run(self, message: str, **kwargs) -> Iterator[RunResponse]:
        """Process the user request and determine which tool to use."""
        try:
            # Initial progress message
            yield RunResponse(content="🔄 **Analyzing your request...**\n\nDetermining the best financial analysis approach...")
            
            response_content = self.determine_and_execute_tool(message)
            
            # Final response
            yield RunResponse(content=str(response_content))
            
        except Exception as e:
            error_message = f"❌ **Error Processing Request**\n\nI encountered an error processing your request: {str(e)}. Please try rephrasing your question."
            yield RunResponse(content=error_message)
        
    def determine_and_execute_tool(self, message: str) -> str:
        """Determine which tool to use based on user message and execute it."""
        message_lower = message.lower()
        
        # Extract assets from the message
        assets = self.extract_assets(message)
        
        # Portfolio Optimization Tools
        if any(kw in message_lower for kw in ["optimize", "portfolio allocation", "weights", "allocation", "optimal"]):
            return self.handle_portfolio_optimization(message, assets)
            
        # Risk Analysis Tools  
        elif any(kw in message_lower for kw in ["risk", "var", "volatility", "drawdown", "cvar", "stress"]):
            return self.handle_risk_analysis(message, assets)
            
        # Data Analysis Tools
        elif any(kw in message_lower for kw in ["analyze data", "correlation", "market data", "trends", "data quality"]):
            return self.handle_data_analysis(message, assets)
            
        # Backtesting Tools
        elif any(kw in message_lower for kw in ["backtest", "strategy", "historical performance", "compare strategies"]):
            return self.handle_backtesting(message, assets)
            
        # Default response with guidance
        return self.provide_guidance(assets)
    
    def extract_assets(self, message: str) -> List[str]:
        """Extract stock symbols from the message."""
        # Match common stock patterns (1-5 uppercase letters, optionally followed by a hyphen and more letters)
        pattern = r'\b[A-Z]{1,5}(?:-[A-Z]+)?\b'
        symbols = re.findall(pattern, message.upper())
        
        # Filter out common words that might match the pattern
        common_words = {'AND', 'OR', 'THE', 'FOR', 'WITH', 'FROM', 'TO', 'IN', 'ON', 'AT', 'BY', 'OF'}
        symbols = [s for s in symbols if s not in common_words]
        
        # If no symbols found, look for asset keywords followed by symbols
        if not symbols:
            # Look for "stocks/assets/portfolio/symbols: X, Y, Z" patterns
            asset_sections = re.search(r'(?:stocks|assets|portfolio|symbols|tickers)[:;]\s*(.*?)(?:$|\.|\n)', message, re.IGNORECASE)
            if asset_sections:
                # Extract potential symbols from the section
                potential_symbols = re.findall(r'\b[A-Za-z]{1,5}(?:-[A-Za-z]+)?\b', asset_sections.group(1))
                symbols = [s.upper() for s in potential_symbols if s.upper() not in common_words]
        
        # Default assets if none found
        if not symbols:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
            
        return symbols[:10]  # Limit to 10 assets max
    
    def handle_portfolio_optimization(self, message: str, assets: List[str]) -> str:
        """Handle portfolio optimization requests using direct tool access."""
        try:
            # Use tools if available, otherwise fallback to workflows
            if TOOLS_AVAILABLE and 'optimize_portfolio' in self.tools:
                # Determine optimization parameters from message
                message_lower = message.lower()
                
                model = "Classic"
                if "black litterman" in message_lower or "bl" in message_lower:
                    model = "BL"
                    
                rm = "MV"  # Default to Mean-Variance
                if "mad" in message_lower:
                    rm = "MAD"
                elif "cvar" in message_lower:
                    rm = "CVaR"
                elif "var" in message_lower and "cvar" not in message_lower:
                    rm = "VaR"
                    
                obj = "MinRisk"  # Default objective
                if "return" in message_lower or "profit" in message_lower:
                    obj = "MaxRet"
                elif "sharpe" in message_lower:
                    obj = "Sharpe"
                
                # Use the direct tool
                assets_str = ",".join(assets)
                result = self.tools['optimize_portfolio'](
                    assets=assets_str,
                    model=model,
                    rm=rm,
                    obj=obj
                )
                
                if result.get("success"):
                    response = f"🎯 **Portfolio Optimization Results**\n\n"
                    response += f"**Assets:** {', '.join(assets)}\n"
                    response += f"**Strategy:** {model} model with {rm} risk measure, {obj} objective\n\n"
                    
                    if "weights" in result:
                        response += "**Optimal Allocation:**\n"
                        for asset, weight in result["weights"].items():
                            response += f"• {asset}: {weight:.2%}\n"
                    
                    if "expected_return" in result:
                        response += f"\n**Expected Annual Return:** {result['expected_return']:.2%}"
                    if "volatility" in result:
                        response += f"\n**Expected Volatility:** {result['volatility']:.2%}"
                    if "sharpe_ratio" in result:
                        response += f"\n**Sharpe Ratio:** {result['sharpe_ratio']:.3f}"
                    
                    return response
                else:
                    return f"❌ **Optimization Failed:** {result.get('error', 'Unknown error')}"
            
            else:
                # Fallback to workflow method
                return self._fallback_portfolio_optimization(message, assets)
                
        except Exception as e:
            return f"❌ **Error in Portfolio Optimization:** {str(e)}"
            message_lower = message.lower()
            
            model = "Classic"
            if "black litterman" in message_lower or "bl" in message_lower:
                model = "BL"
                
            rm = "MV"  # Default to Mean-Variance
            if "mad" in message_lower:
                rm = "MAD"
            elif "cvar" in message_lower or "conditional" in message_lower:
                rm = "CVaR"
                
            obj = "MinRisk"  # Default to minimum risk
            if any(term in message_lower for term in ["sharpe", "return", "maximize", "highest"]):
                obj = "MaxSharpe"
            elif "utility" in message_lower:
                obj = "MaxUtility"
                
            # Use the existing portfolio optimizer
            result_response = self.portfolio_optimizer.optimize_portfolio(
                assets=",".join(assets),
                model=model,
                rm=rm,
                obj=obj
            )
            
            # Extract content from RunResponse
            result = result_response.content if hasattr(result_response, 'content') else result_response
            
            # Format the response
            if result.get("success") and "weights" in result:
                response = f"✅ **Portfolio Optimization Complete**\n\n"
                response += f"**Assets:** {', '.join(assets)}\n"
                response += f"**Method:** {model} model with {rm} risk measure\n"
                response += f"**Objective:** {obj}\n\n"
                response += f"**Optimal Allocation:**\n"
                
                for asset, weight in result["weights"].items():
                    response += f"• {asset}: {weight:.1%}\n"
                
                # Add performance metrics if available
                if "expected_return" in result:
                    response += f"\n**Expected Performance:**\n"
                    response += f"• Expected Return: {result.get('expected_return', 0):.2%}\n"
                    response += f"• Volatility: {result.get('volatility', 0):.2%}\n"
                    response += f"• Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}\n"
                
                response += "\n💡 **Recommendation:** This allocation balances risk and return based on historical data."
                
                return response
            else:
                error_msg = result.get("error", "detailed results are not available") if isinstance(result, dict) else "optimization failed"
                return f"✅ **Portfolio Optimization Complete**\n\nPortfolio optimization completed for {', '.join(assets)}, but {error_msg}."
                
        except Exception as e:
            return f"❌ **Optimization Error**\n\nError optimizing portfolio: {str(e)}"
    
    def handle_risk_analysis(self, message: str, assets: List[str]) -> str:
        """Handle risk analysis requests using direct tool access."""
        try:
            message_lower = message.lower()
            
            # Use tools if available, otherwise fallback to workflows
            if TOOLS_AVAILABLE and ('calculate_risk_metrics' in self.tools or 'stress_test_portfolio' in self.tools):
                
                # Determine if it's a stress test
                if "stress" in message_lower and 'stress_test_portfolio' in self.tools:
                    scenario = "market_crash"  # Default scenario
                    if "recession" in message_lower:
                        scenario = "recession"
                    elif "inflation" in message_lower:
                        scenario = "inflation_shock"
                    elif "interest" in message_lower or "rate" in message_lower:
                        scenario = "interest_rate_hike"
                    
                    assets_str = ",".join(assets)
                    result = self.tools['stress_test_portfolio'](
                        assets=assets_str,
                        scenario=scenario
                    )
                    
                    if result.get("success"):
                        response = f"🎯 **Stress Test Results**\n\n"
                        response += f"**Scenario:** {scenario.replace('_', ' ').title()}\n"
                        response += f"**Assets:** {', '.join(assets)}\n\n"
                        
                        if "stress_results" in result:
                            response += "**Impact Analysis:**\n"
                            for metric, value in result["stress_results"].items():
                                if isinstance(value, (int, float)):
                                    response += f"• {metric.replace('_', ' ').title()}: {value:.2%}\n"
                        
                        return response
                    else:
                        return f"❌ **Stress Test Failed:** {result.get('error', 'Unknown error')}"
                
                # Regular risk metrics
                elif 'calculate_risk_metrics' in self.tools:
                    confidence_level = 0.95
                    if "99%" in message_lower:
                        confidence_level = 0.99
                    elif "90%" in message_lower:
                        confidence_level = 0.90
                    
                    assets_str = ",".join(assets)
                    result = self.tools['calculate_risk_metrics'](
                        assets=assets_str,
                        confidence_level=confidence_level
                    )
                    
                    if result.get("success"):
                        response = f"📊 **Risk Analysis Results**\n\n"
                        response += f"**Assets:** {', '.join(assets)}\n"
                        response += f"**Confidence Level:** {confidence_level:.0%}\n\n"
                        
                        if "risk_metrics" in result:
                            metrics = result["risk_metrics"]
                            response += "**Risk Metrics:**\n"
                            for metric, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    if "var" in metric.lower() or "cvar" in metric.lower():
                                        response += f"• {metric}: {value:.2%}\n"
                                    else:
                                        response += f"• {metric}: {value:.4f}\n"
                        
                        return response
                    else:
                        return f"❌ **Risk Analysis Failed:** {result.get('error', 'Unknown error')}"
            
            else:
                # Fallback to workflow method
                return self._fallback_risk_analysis(message, assets)
                
        except Exception as e:
            return f"❌ **Error in Risk Analysis:** {str(e)}"
    
    def _fallback_risk_analysis(self, message: str, assets: List[str]) -> str:
        """Fallback risk analysis using workflows when tools are not available."""
        # This method handles risk analysis using the original workflow approach
        message_lower = message.lower()
        
        # Determine if it's a stress test
        if "stress" in message_lower:
            scenario = "market_crash"  # Default scenario
            if "recession" in message_lower:
                scenario = "recession"
            elif "inflation" in message_lower:
                scenario = "inflation_shock"
            elif "interest" in message_lower or "rate" in message_lower:
                scenario = "interest_rate_hike"
            
            # Use existing risk analyzer for stress testing
            result_response = self.risk_analyzer.perform_stress_test(
                assets=",".join(assets),
                scenario=scenario
            )
            
            # Extract content from RunResponse
            result = result_response.content if hasattr(result_response, 'content') else result_response
            
            response = f"✅ **Stress Test Complete**\n\n"
            response += f"**Scenario:** {scenario.replace('_', ' ').title()}\n"
            response += f"**Assets:** {', '.join(assets)}\n\n"
            
            if result.get("success") and "results" in result:
                stress_results = result["results"]
                response += f"**Stress Test Impact:**\n"
                for metric, value in stress_results.items():
                    if isinstance(value, (int, float)):
                        response += f"• {metric.replace('_', ' ').title()}: {value:.2%}\n"
                
                response += f"\n💡 **Analysis:** This shows how your portfolio might perform under {scenario.replace('_', ' ')} conditions."
            else:
                error_msg = result.get("error", "stress test failed") if isinstance(result, dict) else "analysis failed"
                response += f"**Status:** {error_msg}"
            
            return response
        else:
            # Regular risk metrics
            result_response = self.risk_analyzer.calculate_risk_metrics(
                assets=",".join(assets)
            )
            
            # Extract content from RunResponse
            result = result_response.content if hasattr(result_response, 'content') else result_response
            
            response = f"✅ **Risk Analysis Complete**\n\n"
            response += f"**Assets:** {', '.join(assets)}\n\n"
            
            if result.get("success") and "metrics" in result:
                metrics = result["metrics"]
                response += f"**Key Risk Metrics:**\n"
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        response += f"• {metric.replace('_', ' ').title()}: {value:.2%}\n"
                
                response += f"\n💡 **Risk Assessment:** These metrics help evaluate your portfolio's risk profile and downside potential."
            else:
                error_msg = result.get("error", "risk analysis failed") if isinstance(result, dict) else "analysis failed"
                response += f"**Status:** {error_msg}"
            
            return response
    
    def handle_data_analysis(self, message: str, assets: List[str]) -> str:
        """Handle data analysis requests."""
        try:
            message_lower = message.lower()
            
            # Determine analysis type
            analysis_type = "overview"
            if "correlation" in message_lower:
                analysis_type = "correlation"
            elif "quality" in message_lower:
                analysis_type = "quality"
            elif "trend" in message_lower:
                analysis_type = "trend"
            
            # Use existing data analyzer
            result_response = self.data_analyzer.analyze_market_data(
                assets=",".join(assets),
                analysis_type=analysis_type
            )
            
            # Extract content from RunResponse
            result = result_response.content if hasattr(result_response, 'content') else result_response
            
            response = f"📈 **Market Data Analysis**\n\n"
            response += f"**Assets:** {', '.join(assets)}\n"
            response += f"**Analysis Type:** {analysis_type.title()}\n\n"
            
            if result.get("success") and "analysis" in result:
                analysis = result["analysis"]
                response += f"**Results:**\n"
                
                if analysis_type == "correlation" and "correlation_matrix" in analysis:
                    response += f"**Correlation Analysis:**\n"
                    # Add correlation insights
                    response += f"• Analysis shows correlation patterns between assets\n"
                    
                elif analysis_type == "quality" and "data_quality" in analysis:
                    response += f"**Data Quality Check:**\n"
                    quality = analysis["data_quality"]
                    for asset in assets:
                        if asset in quality:
                            response += f"• {asset}: {quality[asset].get('completeness', 'N/A')}\n"
                
                elif "summary" in analysis:
                    summary = analysis["summary"]
                    for key, value in summary.items():
                        if isinstance(value, (int, float)):
                            response += f"• {key.replace('_', ' ').title()}: {value:.2%}\n"
                        else:
                            response += f"• {key.replace('_', ' ').title()}: {value}\n"
            else:
                error_msg = result.get("error", "data analysis failed") if isinstance(result, dict) else "analysis failed"
                response += f"**Status:** {error_msg}"
            
            return response
            
        except Exception as e:
            return f"Error analyzing market data: {str(e)}"
    
    def handle_backtesting(self, message: str, assets: List[str]) -> str:
        """Handle backtesting requests."""
        try:
            message_lower = message.lower()
            
            # Determine strategy type
            strategy = "equal_weight"
            if "momentum" in message_lower:
                strategy = "momentum"
            elif "mean" in message_lower and "var" in message_lower:
                strategy = "mean_variance"
            elif "risk" in message_lower and "par" in message_lower:
                strategy = "risk_parity"
            
            # Determine time period
            period = "1Y"
            if "6m" in message_lower or "6 month" in message_lower:
                period = "6M"
            elif "2y" in message_lower or "2 year" in message_lower:
                period = "2Y"
            elif "3y" in message_lower or "3 year" in message_lower:
                period = "3Y"
            
            # Use existing portfolio backtester
            result_response = self.portfolio_backtester.backtest_strategy(
                assets=",".join(assets),
                strategy=strategy,
                period=period
            )
            
            # Extract content from RunResponse
            result = result_response.content if hasattr(result_response, 'content') else result_response
            
            response = f"🔄 **Backtesting Results**\n\n"
            response += f"**Strategy:** {strategy.replace('_', ' ').title()}\n"
            response += f"**Assets:** {', '.join(assets)}\n"
            response += f"**Period:** {period}\n\n"
            
            if result.get("success") and "performance" in result:
                perf = result["performance"]
                response += f"**Performance Metrics:**\n"
                
                for metric, value in perf.items():
                    if isinstance(value, (int, float)):
                        if "return" in metric.lower():
                            response += f"• {metric.replace('_', ' ').title()}: {value:.2%}\n"
                        elif "ratio" in metric.lower():
                            response += f"• {metric.replace('_', ' ').title()}: {value:.2f}\n"
                        else:
                            response += f"• {metric.replace('_', ' ').title()}: {value:.2%}\n"
            else:
                error_msg = result.get("error", "backtesting failed") if isinstance(result, dict) else "analysis failed"
                response += f"**Status:** {error_msg}"
            
            return response
            
        except Exception as e:
            return f"Error backtesting strategy: {str(e)}"
    
    def provide_guidance(self, assets: List[str]) -> str:
        """Provide guidance when the request is unclear."""
        response = f"👋 **Welcome to the Financial Analysis Team!**\n\n"
        
        if assets and assets != ["AAPL", "MSFT", "GOOGL", "AMZN"]:
            response += f"I detected these assets: {', '.join(assets)}\n\n"
        
        response += f"I can help you with:\n\n"
        response += f"🎯 **Portfolio Optimization**\n"
        response += f"• 'Optimize a portfolio with AAPL, MSFT, GOOGL'\n"
        response += f"• 'Create minimum risk allocation for tech stocks'\n\n"
        
        response += f"📊 **Risk Analysis**\n"
        response += f"• 'Calculate risk metrics for my portfolio'\n"
        response += f"• 'Stress test AAPL and TSLA portfolio'\n\n"
        
        response += f"📈 **Data Analysis**\n"
        response += f"• 'Analyze correlation between AAPL and MSFT'\n"
        response += f"• 'Check data quality for tech stocks'\n\n"
        
        response += f"🔄 **Strategy Backtesting**\n"
        response += f"• 'Backtest momentum strategy with these assets'\n"
        response += f"• 'Compare equal weight vs optimized strategies'\n\n"
        
        response += f"Just tell me what you'd like to analyze and I'll use the appropriate tool!"
        
        return response
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Get information about all available tools."""
        if TOOLS_AVAILABLE:
            try:
                # Use the get_tool_info function if available
                return get_tool_info()
            except:
                # Fallback to basic tool information
                return {
                    "categories": {
                        "Portfolio Optimization": list(filter(lambda x: 'optimize' in x or 'efficient' in x or 'sharpe' in x, self.tools.keys())),
                        "Risk Analysis": list(filter(lambda x: 'risk' in x or 'stress' in x, self.tools.keys())),
                        "Market Analysis": list(filter(lambda x: 'analyze' in x or 'correlation' in x, self.tools.keys())),
                        "Strategy Testing": list(filter(lambda x: 'backtest' in x, self.tools.keys())),
                        "Reporting": list(filter(lambda x: 'report' in x or 'validate' in x, self.tools.keys()))
                    },
                    "total_tools": len(self.tools),
                    "tools_available": True
                }
        else:
            return {
                "categories": {
                    "Workflows": ["Portfolio Optimizer", "Risk Analyzer", "Data Analyzer", "Portfolio Backtester"]
                },
                "total_tools": 4,
                "tools_available": False
            }
    
    def call_tool_directly(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a specific tool directly by name."""
        if TOOLS_AVAILABLE and tool_name in self.tools:
            try:
                return self.tools[tool_name](**kwargs)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error calling {tool_name}: {str(e)}"
                }
        else:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not available. Available tools: {list(self.tools.keys()) if self.tools else 'None'}"
            }
