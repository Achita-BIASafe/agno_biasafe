# BiasafeAI Finance Tools - Team-Based Architecture

## Overview

This restructure transforms the AgnoAI framework from individual agent workflows into a unified **team-based architecture** where each financial analysis function is available as an individual tool that can be called directly by LLMs.

## Architecture Changes

### Before (Individual Agents)
```
- portfolio_optimizer_agent.py  → Single purpose agent
- risk_analysis_agent.py        → Single purpose agent  
- data_analysis_agent.py        → Single purpose agent
- portfolio_backtesting_agent.py → Single purpose agent
```

### After (Team-Based Tools)
```
- finance_team.py      → Unified agent that routes to appropriate tools
- finance_tools.py     → Registry of individual callable functions  
- finance_client.py    → Unified interface for both direct and conversational access
```

## Key Components

### 1. FinanceTeam Agent (`finance_team.py`)
- **Purpose**: Single conversational agent that analyzes requests and routes to appropriate tools
- **Features**:
  - Natural language request parsing
  - Asset extraction from text
  - Intelligent tool selection and routing
  - Conversational interface for complex multi-step analysis

### 2. Finance Tools Registry (`finance_tools.py`)
- **Purpose**: Collection of individual callable financial analysis tools
- **Available Tools**:
  - `optimize_portfolio()` - Portfolio optimization with various objectives
  - `calculate_risk_metrics()` - Risk analysis and VaR calculations
  - `backtest_strategy()` - Strategy backtesting and performance analysis
  - `analyze_market_data()` - Market data analysis and trends
  - `get_performance_metrics()` - Performance measurement and attribution
  - `analyze_correlations()` - Correlation analysis between assets
  - `generate_reports()` - Automated report generation
  - `plot_data()` - Data visualization and charting

### 3. Finance Client (`finance_client.py`)
- **Purpose**: Unified client interface providing both direct tool access and conversational interaction
- **Features**:
  - Direct tool access via specialized interfaces (PortfolioTools, RiskTools, etc.)
  - Conversational chat interface through FinanceTeam agent
  - Tool discovery and documentation
  - Status monitoring and health checks

## Usage Patterns

### Direct Tool Access (For LLMs)
```python
from finance_client import create_client

client = create_client()

# Portfolio optimization
result = client.portfolio.optimize_portfolio(assets="AAPL,MSFT,GOOGL")

# Risk analysis  
risk_metrics = client.risk.calculate_risk_metrics(assets="AAPL,MSFT,GOOGL")

# Backtesting
backtest_results = client.backtest.backtest_strategy(assets="AAPL,MSFT,GOOGL")
```

### Conversational Interface (For Users)
```python
client = create_client()

# Natural language requests
response = client.chat("Optimize a portfolio with Apple, Microsoft, and Google")
response = client.chat("What are the risks of holding equal weights in tech stocks?")
response = client.chat("Backtest a momentum strategy over the past year")
```

### Tool Discovery
```python
# List all available tools
tools = client.list_tools()

# Get tool information
info = client.get_tool_info('optimize_portfolio')

# Get tool function directly
tool_func = client.get_tool('backtest_strategy')
```

## Benefits of New Architecture

### For LLMs
1. **Granular Tool Selection**: LLMs can call specific financial functions directly based on user intent
2. **Standardized Interface**: All tools follow consistent input/output patterns
3. **Composable Operations**: Tools can be chained together for complex analysis workflows
4. **Clear Documentation**: Each tool has detailed documentation and parameter specifications

### For Users
1. **Conversational Interface**: Natural language interaction through FinanceTeam agent
2. **Unified Access**: Single client interface for all financial analysis needs
3. **Flexible Usage**: Choose between direct tool access or conversational interaction
4. **Backward Compatibility**: Existing workflows still available during transition

### For Developers
1. **Modular Design**: Individual tools can be modified or extended independently
2. **Easy Testing**: Each tool can be tested in isolation
3. **Clear Separation**: Business logic separated from interaction layer
4. **Extensible**: New tools can be easily added to the registry

## File Structure

```
agno_biasafe/
├── finance_team.py          # Unified conversational agent
├── finance_tools.py         # Individual tool registry
├── finance_client.py        # Client interface
├── finance_demo.py          # Standalone demo script
├── new_playground.py        # Updated playground with team architecture
├── test_architecture.py     # Architecture validation tests
│
├── [Legacy Files - Backward Compatibility]
├── portfolio_optimizer.py      # Original workflow
├── risk_analyzer.py           # Original workflow  
├── portfolio_backtester.py    # Original workflow
├── data_analyzer.py           # Original workflow
├── plotting_tools.py          # Original workflow
├── report_generator.py        # Original workflow
├── data_utilities.py          # Original workflow
└── playground.py              # Original playground
```

## Tool Categories

### Portfolio Management
- `optimize_portfolio` - Portfolio optimization using various objectives (Sharpe, min volatility, etc.)
- `rebalance_portfolio` - Portfolio rebalancing and weight adjustment
- `calculate_efficient_frontier` - Efficient frontier calculation and visualization

### Risk Analysis  
- `calculate_risk_metrics` - Comprehensive risk metrics (VaR, CVaR, volatility, etc.)
- `stress_test_portfolio` - Stress testing under various market scenarios
- `analyze_drawdowns` - Drawdown analysis and recovery periods

### Performance Analysis
- `get_performance_metrics` - Performance attribution and measurement
- `benchmark_comparison` - Performance comparison against benchmarks
- `factor_analysis` - Factor exposure and attribution analysis

### Market Analysis
- `analyze_market_data` - Market data analysis and trend identification  
- `analyze_correlations` - Correlation analysis between assets and markets
- `sector_analysis` - Sector allocation and performance analysis

### Backtesting
- `backtest_strategy` - Strategy backtesting with various parameters
- `walk_forward_analysis` - Out-of-sample testing and validation
- `monte_carlo_simulation` - Monte Carlo simulation for robustness testing

### Reporting & Visualization
- `generate_reports` - Automated report generation in various formats
- `plot_data` - Data visualization and charting
- `create_dashboards` - Interactive dashboard creation

## Next Steps

1. **Install Dependencies**: Install required packages (agno, riskfolio-lib, etc.)
2. **Configure Environment**: Set up API keys and database connections
3. **Test Integration**: Validate all tools work with real market data
4. **LLM Integration**: Test with LLM frameworks to ensure proper tool calling
5. **Performance Optimization**: Optimize tool execution for production use

## Migration Guide

For existing users:
1. Existing workflows continue to work unchanged
2. New team-based tools provide enhanced functionality
3. Gradual migration possible by adopting new client interface
4. Full backward compatibility maintained during transition period

This architecture enables LLMs to make intelligent decisions about which specific financial analysis tools to use while maintaining a user-friendly conversational interface for complex financial analysis workflows.
