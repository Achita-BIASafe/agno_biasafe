"""
Workflows Package for Agno BIASafe AI

This package contains all workflow and agent classes that extend the Agno framework
for financial analysis and portfolio management.

Workflows (extend Workflow class):
- PortfolioOptimizer: Portfolio optimization using various risk measures
- RiskAnalyzer: Risk analysis and measurement tools
- PortfolioBacktester: Backtesting portfolio strategies
- DataAnalyzer: Data analysis and processing tools
- PlottingTools: Visualization and plotting utilities
- ReportGenerator: Financial report generation
- DataUtilities: Data utilities and helper functions

Agents (extend Agent class):
- Portfolio Optimizer Agent: Conversational portfolio optimization
- Risk Analysis Agent: Conversational risk analysis
- Data Analysis Agent: Conversational data analysis
- Portfolio Backtesting Agent: Conversational backtesting

These workflows handle the core financial analysis functionality and can be called
by tools or used directly by agents.
"""

# Import all workflow classes
from .portfolio_optimizer import PortfolioOptimizer
from .risk_analyzer import RiskAnalyzer
from .portfolio_backtester import PortfolioBacktester
from .data_analyzer import DataAnalyzer
from .plotting_tools import PlottingTools
from .report_generator import ReportGenerator
from .data_utilities import DataUtilities

# Import all agent classes
from .portfolio_optimizer_agent import PortfolioOptimizerAgent
from .risk_analysis_agent import RiskAnalysisAgent
from .data_analysis_agent import DataAnalysisAgent
from .portfolio_backtesting_agent import PortfolioBacktestingAgent

__all__ = [
    # Workflows
    'PortfolioOptimizer',
    'RiskAnalyzer', 
    'PortfolioBacktester',
    'DataAnalyzer',
    'PlottingTools',
    'ReportGenerator',
    'DataUtilities',
    # Agents
    'PortfolioOptimizerAgent',
    'RiskAnalysisAgent',
    'DataAnalysisAgent', 
    'PortfolioBacktestingAgent'
]
