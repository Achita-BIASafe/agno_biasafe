"""
Financial Analysis Tools Registry

This module provides individual tools that can be called directly by LLMs
or accessed through the finance team interface.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import requests
import re
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from api_config import get_endpoint

# Import existing workflow components
from workflows.portfolio_optimizer import PortfolioOptimizer
from workflows.risk_analyzer import RiskAnalyzer
from workflows.data_analyzer import DataAnalyzer
from workflows.portfolio_backtester import PortfolioBacktester

# Initialize workflow components
_portfolio_optimizer = PortfolioOptimizer()
_risk_analyzer = RiskAnalyzer()
_data_analyzer = DataAnalyzer()
_portfolio_backtester = PortfolioBacktester()

# Tool registry
finance_tools = {}

# ============================================================================
# PORTFOLIO OPTIMIZATION TOOLS
# ============================================================================

def optimize_portfolio(
    assets: str,
    model: str = "Classic", 
    rm: str = "MV", 
    obj: str = "MinRisk",
    rf: float = 0.0,
    max_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Optimize a portfolio based on modern portfolio theory.
    
    Args:
        assets: Comma-separated list of stock symbols (e.g., "AAPL,MSFT,GOOGL")
        model: Optimization model ("Classic", "BL" for Black-Litterman, "FM" for Factor Model)
        rm: Risk measure ("MV" for variance, "MAD" for mean absolute deviation, "CVaR")
        obj: Objective function ("MinRisk", "MaxSharpe", "MaxReturn", "Utility")
        rf: Risk-free rate (annualized)
        max_weight: Maximum weight constraint per asset (0.0 to 1.0)
        
    Returns:
        Dictionary with optimized portfolio weights and performance metrics
    """
    try:
        # Use existing portfolio optimizer
        result = _portfolio_optimizer.optimize_portfolio(
            assets=assets,
            model=model,
            rm=rm,
            obj=obj,
            rf=rf
        )
        
        return {
            "success": True,
            "tool": "optimize_portfolio",
            "assets": assets.split(","),
            "parameters": {
                "model": model,
                "risk_measure": rm,
                "objective": obj,
                "risk_free_rate": rf,
                "max_weight": max_weight
            },
            "results": result
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "optimize_portfolio",
            "error": str(e),
            "error_type": type(e).__name__
        }

def create_efficient_frontier(
    assets: str,
    model: str = "Classic",
    rm: str = "MV",
    points: int = 50
) -> Dict[str, Any]:
    """
    Generate efficient frontier for a set of assets.
    
    Args:
        assets: Comma-separated list of stock symbols
        model: Optimization model to use
        rm: Risk measure
        points: Number of points on the frontier
        
    Returns:
        Dictionary with efficient frontier data points
    """
    try:
        # Use existing portfolio optimizer
        result = _portfolio_optimizer.generate_efficient_frontier(
            assets=assets,
            model=model,
            rm=rm,
            points=points
        )
        
        return {
            "success": True,
            "tool": "create_efficient_frontier",
            "assets": assets.split(","),
            "parameters": {
                "model": model,
                "risk_measure": rm,
                "points": points
            },
            "results": result
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "create_efficient_frontier",
            "error": str(e),
            "error_type": type(e).__name__
        }

# Register portfolio tools
finance_tools["optimize_portfolio"] = optimize_portfolio
finance_tools["create_efficient_frontier"] = create_efficient_frontier

# ============================================================================
# RISK ANALYSIS TOOLS
# ============================================================================

def calculate_risk_metrics(
    assets: str,
    portfolio_weights: str = "",
    risk_measure: str = "all",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate comprehensive risk metrics for a portfolio.
    
    Args:
        assets: Comma-separated list of stock symbols
        portfolio_weights: Optional comma-separated weights (e.g., "0.3,0.3,0.4")
        risk_measure: Risk measure to focus on ("VaR", "CVaR", "Volatility", "all")
        confidence_level: Confidence level for VaR/CVaR calculations
    
    Returns:
        Dictionary with calculated risk metrics
    """
    try:
        # Use existing risk analyzer
        result = _risk_analyzer.calculate_risk_metrics(
            assets=assets,
            weights=portfolio_weights if portfolio_weights else None
        )
        
        return {
            "success": True,
            "tool": "calculate_risk_metrics",
            "assets": assets.split(","),
            "parameters": {
                "portfolio_weights": portfolio_weights,
                "risk_measure": risk_measure,
                "confidence_level": confidence_level
            },
            "results": result
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "calculate_risk_metrics",
            "error": str(e),
            "error_type": type(e).__name__
        }

def stress_test_portfolio(
    assets: str,
    portfolio_weights: str = "",
    scenario: str = "market_crash"
) -> Dict[str, Any]:
    """
    Perform stress testing on a portfolio under different market scenarios.
    
    Args:
        assets: Comma-separated list of stock symbols
        portfolio_weights: Optional comma-separated weights
        scenario: Market scenario ("market_crash", "recession", "inflation_shock", "interest_rate_hike")
    
    Returns:
        Dictionary with stress test results
    """
    try:
        # Use existing risk analyzer
        result = _risk_analyzer.perform_stress_test(
            assets=assets,
            scenario=scenario,
            weights=portfolio_weights if portfolio_weights else None
        )
        
        return {
            "success": True,
            "tool": "stress_test_portfolio",
            "assets": assets.split(","),
            "parameters": {
                "portfolio_weights": portfolio_weights,
                "scenario": scenario
            },
            "results": result
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "stress_test_portfolio",
            "error": str(e),
            "error_type": type(e).__name__
        }

def calculate_var_cvar(
    assets: str,
    portfolio_weights: str = "",
    confidence_level: float = 0.95,
    method: str = "historical"
) -> Dict[str, Any]:
    """
    Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    
    Args:
        assets: Comma-separated list of stock symbols
        portfolio_weights: Optional comma-separated weights
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: Calculation method ("historical", "parametric", "monte_carlo")
    
    Returns:
        Dictionary with VaR and CVaR calculations
    """
    try:
        # Use existing risk analyzer
        result = _risk_analyzer.calculate_var_cvar(
            assets=assets,
            weights=portfolio_weights if portfolio_weights else None,
            confidence_level=confidence_level,
            method=method
        )
        
        return {
            "success": True,
            "tool": "calculate_var_cvar",
            "assets": assets.split(","),
            "parameters": {
                "portfolio_weights": portfolio_weights,
                "confidence_level": confidence_level,
                "method": method
            },
            "results": result
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "calculate_var_cvar",
            "error": str(e),
            "error_type": type(e).__name__
        }

# Register risk analysis tools
finance_tools["calculate_risk_metrics"] = calculate_risk_metrics
finance_tools["stress_test_portfolio"] = stress_test_portfolio
finance_tools["calculate_var_cvar"] = calculate_var_cvar

# ============================================================================
# DATA ANALYSIS TOOLS
# ============================================================================

def analyze_market_data(
    assets: str,
    analysis_type: str = "overview",
    time_period: str = "1y"
) -> Dict[str, Any]:
    """
    Analyze market data for specific assets.
    
    Args:
        assets: Comma-separated list of stock symbols
        analysis_type: Type of analysis ("correlation", "trend", "quality_check", "overview")
        time_period: Time period for analysis ("1m", "3m", "6m", "1y", "3y", "5y")
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Use existing data analyzer
        result = _data_analyzer.analyze_market_data(
            assets=assets,
            analysis_type=analysis_type
        )
        
        return {
            "success": True,
            "tool": "analyze_market_data",
            "assets": assets.split(","),
            "parameters": {
                "analysis_type": analysis_type,
                "time_period": time_period
            },
            "results": result
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "analyze_market_data",
            "error": str(e),
            "error_type": type(e).__name__
        }

def calculate_correlation_matrix(
    assets: str,
    method: str = "pearson",
    time_period: str = "1y"
) -> Dict[str, Any]:
    """
    Calculate correlation matrix between assets.
    
    Args:
        assets: Comma-separated list of stock symbols
        method: Correlation method ("pearson", "spearman", "kendall")
        time_period: Time period for analysis
        
    Returns:
        Dictionary with correlation matrix and analysis
    """
    try:
        # Use existing data analyzer
        result = _data_analyzer.calculate_correlation_matrix(
            assets=assets,
            method=method
        )
        
        return {
            "success": True,
            "tool": "calculate_correlation_matrix",
            "assets": assets.split(","),
            "parameters": {
                "method": method,
                "time_period": time_period
            },
            "results": result
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "calculate_correlation_matrix",
            "error": str(e),
            "error_type": type(e).__name__
        }

def check_data_quality(
    assets: str,
    checks: str = "all"
) -> Dict[str, Any]:
    """
    Check data quality for specified assets.
    
    Args:
        assets: Comma-separated list of stock symbols
        checks: Types of checks to perform ("missing", "outliers", "gaps", "all")
        
    Returns:
        Dictionary with data quality assessment
    """
    try:
        # Use existing data analyzer
        result = _data_analyzer.check_data_quality(
            assets=assets
        )
        
        return {
            "success": True,
            "tool": "check_data_quality",
            "assets": assets.split(","),
            "parameters": {
                "checks": checks
            },
            "results": result
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "check_data_quality",
            "error": str(e),
            "error_type": type(e).__name__
        }

# Register data analysis tools
finance_tools["analyze_market_data"] = analyze_market_data
finance_tools["calculate_correlation_matrix"] = calculate_correlation_matrix
finance_tools["check_data_quality"] = check_data_quality

# ============================================================================
# BACKTESTING TOOLS
# ============================================================================

def backtest_strategy(
    assets: str,
    strategy: str = "equal_weight",
    period: str = "1y",
    rebalance: str = "monthly",
    benchmark: str = ""
) -> Dict[str, Any]:
    """
    Backtest an investment strategy on historical data.
    
    Args:
        assets: Comma-separated list of stock symbols
        strategy: Strategy to backtest ("equal_weight", "mean_variance", "risk_parity", "momentum")
        period: Backtest period ("6m", "1y", "3y", "5y")
        rebalance: Rebalancing frequency ("weekly", "monthly", "quarterly", "yearly")
        benchmark: Benchmark symbol for comparison (optional)
        
    Returns:
        Dictionary with backtesting results
    """
    try:
        # Use existing portfolio backtester
        result = _portfolio_backtester.backtest_strategy(
            assets=assets,
            strategy=strategy,
            period=period
        )
        
        return {
            "success": True,
            "tool": "backtest_strategy",
            "assets": assets.split(","),
            "parameters": {
                "strategy": strategy,
                "period": period,
                "rebalance": rebalance,
                "benchmark": benchmark
            },
            "results": result
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "backtest_strategy",
            "error": str(e),
            "error_type": type(e).__name__
        }

def compare_strategies(
    assets: str,
    strategies: str = "equal_weight,mean_variance",
    period: str = "1y",
    rebalance: str = "monthly"
) -> Dict[str, Any]:
    """
    Compare multiple investment strategies side by side.
    
    Args:
        assets: Comma-separated list of stock symbols
        strategies: Comma-separated list of strategies to compare
        period: Backtest period
        rebalance: Rebalancing frequency
        
    Returns:
        Dictionary with strategy comparison results
    """
    try:
        # Parse strategies
        strategy_list = [s.strip() for s in strategies.split(",")]
        
        # Run backtests for each strategy
        comparison_results = {}
        for strategy in strategy_list:
            result = backtest_strategy(assets, strategy, period, rebalance)
            if result["success"]:
                comparison_results[strategy] = result["results"]
            else:
                return result  # Return the error
        
        return {
            "success": True,
            "tool": "compare_strategies",
            "assets": assets.split(","),
            "parameters": {
                "strategies": strategy_list,
                "period": period,
                "rebalance": rebalance
            },
            "results": {
                "individual_results": comparison_results,
                "comparison_summary": _create_strategy_comparison(comparison_results)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "compare_strategies",
            "error": str(e),
            "error_type": type(e).__name__
        }

def analyze_drawdowns(
    assets: str,
    portfolio_weights: str = "",
    period: str = "1y"
) -> Dict[str, Any]:
    """
    Analyze portfolio drawdowns over time.
    
    Args:
        assets: Comma-separated list of stock symbols
        portfolio_weights: Optional comma-separated weights
        period: Analysis period
        
    Returns:
        Dictionary with drawdown analysis
    """
    try:
        # Use existing portfolio backtester
        result = _portfolio_backtester.analyze_drawdowns(
            assets=assets,
            weights=portfolio_weights if portfolio_weights else None,
            period=period
        )
        
        return {
            "success": True,
            "tool": "analyze_drawdowns",
            "assets": assets.split(","),
            "parameters": {
                "portfolio_weights": portfolio_weights,
                "period": period
            },
            "results": result
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "analyze_drawdowns",
            "error": str(e),
            "error_type": type(e).__name__
        }

# Register backtesting tools
finance_tools["backtest_strategy"] = backtest_strategy
finance_tools["compare_strategies"] = compare_strategies
finance_tools["analyze_drawdowns"] = analyze_drawdowns

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _create_strategy_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary comparison of multiple strategies."""
    if not results:
        return {}
    
    # Extract key metrics for comparison
    strategies = list(results.keys())
    comparison = {}
    
    # Compare key metrics
    metrics = ["total_return", "annualized_return", "volatility", "sharpe_ratio", "max_drawdown"]
    
    for metric in metrics:
        comparison[metric] = {}
        for strategy in strategies:
            if "performance" in results[strategy] and metric in results[strategy]["performance"]:
                comparison[metric][strategy] = results[strategy]["performance"][metric]
    
    # Determine best strategy for each metric
    best_strategies = {}
    for metric in metrics:
        if comparison[metric]:
            if metric in ["volatility", "max_drawdown"]:  # Lower is better
                best_strategies[metric] = min(comparison[metric].items(), key=lambda x: x[1])
            else:  # Higher is better
                best_strategies[metric] = max(comparison[metric].items(), key=lambda x: x[1])
    
    return {
        "metric_comparison": comparison,
        "best_performers": best_strategies,
        "overall_ranking": _rank_strategies(results)
    }

def _rank_strategies(results: Dict[str, Any]) -> List[str]:
    """Rank strategies based on risk-adjusted returns (Sharpe ratio)."""
    rankings = []
    
    for strategy, result in results.items():
        if "performance" in result and "sharpe_ratio" in result["performance"]:
            sharpe = result["performance"]["sharpe_ratio"]
            rankings.append((strategy, sharpe))
    
    # Sort by Sharpe ratio (descending)
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    return [strategy for strategy, _ in rankings]

# ============================================================================
# TOOL INFORMATION AND REGISTRY
# ============================================================================

def get_tool_info(tool_name: str = None) -> Dict[str, Any]:
    """
    Get information about available tools.
    
    Args:
        tool_name: Specific tool name to get info for (optional)
        
    Returns:
        Dictionary with tool information
    """
    tool_categories = {
        "Portfolio Optimization": [
            "optimize_portfolio",
            "create_efficient_frontier"
        ],
        "Risk Analysis": [
            "calculate_risk_metrics",
            "stress_test_portfolio", 
            "calculate_var_cvar"
        ],
        "Data Analysis": [
            "analyze_market_data",
            "calculate_correlation_matrix",
            "check_data_quality"
        ],
        "Backtesting": [
            "backtest_strategy",
            "compare_strategies",
            "analyze_drawdowns"
        ]
    }
    
    if tool_name:
        if tool_name in finance_tools:
            func = finance_tools[tool_name]
            return {
                "name": tool_name,
                "description": func.__doc__,
                "available": True
            }
        else:
            return {
                "name": tool_name,
                "available": False,
                "error": "Tool not found"
            }
    else:
        return {
            "categories": tool_categories,
            "total_tools": len(finance_tools),
            "available_tools": list(finance_tools.keys())
        }

def list_tools() -> List[str]:
    """Get a list of all available tool names."""
    return list(finance_tools.keys())

def get_tool(tool_name: str):
    """Get a specific tool function by name."""
    return finance_tools.get(tool_name)

# Add utility functions to registry
finance_tools["get_tool_info"] = get_tool_info
finance_tools["list_tools"] = list_tools
