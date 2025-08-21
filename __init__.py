"""
AgnoRiskfolio - A Python client for the Riskfolio Engine API.

This package provides a comprehensive client for interacting with the Riskfolio Engine API,
enabling portfolio optimization, risk analysis, data processing, and more.
"""

from .main import AgnoRiskfolio, create_client
from .portfolio import PortfolioOptimizer
from .risk import RiskAnalyzer
from .data import DataHandler
from .plotting import Plotter
from .reports import ReportGenerator
from .client import RiskfolioClient
from .config import config

# Import agents for conversational AI
try:
    from .portfolio_optimizer_agent import PortfolioOptimizerAgent
    from .risk_analysis_agent import RiskAnalysisAgent
    from .data_analysis_agent import DataAnalysisAgent
    from .portfolio_backtesting_agent import PortfolioBacktestingAgent
    
    # Add agents to exports
    __all__ = [
        "AgnoRiskfolio",
        "create_client",
        "PortfolioOptimizer",
        "RiskAnalyzer",
        "DataHandler",
        "Plotter",
        "ReportGenerator",
        "RiskfolioClient",
        "config",
        "PortfolioOptimizerAgent",
        "RiskAnalysisAgent", 
        "DataAnalysisAgent",
        "PortfolioBacktestingAgent"
    ]
except ImportError:
    # Fallback if agent imports fail
    __all__ = [
        "AgnoRiskfolio",
        "create_client",
        "PortfolioOptimizer",
        "RiskAnalyzer",
        "DataHandler",
        "Plotter",
        "ReportGenerator",
        "RiskfolioClient",
        "config"
    ]

__version__ = "0.1.0"
