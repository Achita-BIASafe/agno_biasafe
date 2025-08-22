"""Tools package for BiasafeAI Agno Framework

This package provides individual financial analysis tools that can be called
directly by LLMs or accessed through conversational interfaces.
"""

from .finance_tools import (
    finance_tools,
    optimize_portfolio,
    create_efficient_frontier,
    calculate_risk_metrics,
    stress_test_portfolio,
    calculate_var_cvar,
    analyze_market_data,
    calculate_correlation_matrix,
    check_data_quality,
    backtest_strategy,
    compare_strategies,
    analyze_drawdowns,
    get_tool_info,
    list_tools,
    get_tool
)

from .finance_team import FinanceTeam
from .finance_client import FinanceClient, create_client

__all__ = [
    # Individual tools
    'finance_tools',
    'optimize_portfolio',
    'create_efficient_frontier',
    'calculate_risk_metrics',
    'stress_test_portfolio',
    'calculate_var_cvar',
    'analyze_market_data',
    'calculate_correlation_matrix',
    'check_data_quality',
    'backtest_strategy',
    'compare_strategies',
    'analyze_drawdowns',
    'get_tool_info',
    'list_tools',
    'get_tool',
    
    # Team and client interfaces
    'FinanceTeam',
    'FinanceClient',
    'create_client'
]
