# Tools Directory - Financial Analysis Tools

## Overview
The `tools/` directory contains all the individual financial analysis tools that can be called directly by LLMs or accessed through conversational interfaces.

## Structure

### Individual API-Calling Tools (`finance_tools.py`)
Each tool calls specific APIs through the workflow components:

#### Portfolio Optimization Tools:
- `optimize_portfolio()` - Optimize portfolio weights using various risk measures
- `create_efficient_frontier()` - Generate efficient frontier for asset sets

#### Risk Analysis Tools:
- `calculate_risk_metrics()` - Calculate VaR, CVaR, volatility, and other risk metrics
- `stress_test_portfolio()` - Stress test portfolios under market scenarios
- `calculate_var_cvar()` - Calculate Value at Risk and Conditional VaR

#### Data Analysis Tools:
- `analyze_market_data()` - Analyze market data and trends
- `calculate_correlation_matrix()` - Calculate asset correlation matrices
- `check_data_quality()` - Check data quality and completeness

#### Backtesting Tools:
- `backtest_strategy()` - Backtest investment strategies
- `compare_strategies()` - Compare multiple strategies side-by-side
- `analyze_drawdowns()` - Analyze portfolio drawdown patterns

### Team Agent (`finance_team.py`)
- `FinanceTeam` class - Unified conversational agent that routes requests to appropriate tools
- Supports natural language interaction
- Determines which tool to use based on user intent
- Formats responses with rich financial insights

### Client Interface (`finance_client.py`)
- `FinanceClient` class - Unified interface for accessing all capabilities
- Structured access through specialized interfaces:
  - `client.portfolio.*` - Portfolio optimization tools
  - `client.risk.*` - Risk analysis tools  
  - `client.data.*` - Data analysis tools
  - `client.backtest.*` - Backtesting tools
- Conversational access through `client.chat()` method

## Usage Examples

### Direct Tool Access
```python
from tools import optimize_portfolio, calculate_risk_metrics

# Direct API calls
result = optimize_portfolio(assets="AAPL,MSFT,GOOGL", obj="MaxSharpe")
risk = calculate_risk_metrics(assets="AAPL,MSFT,GOOGL")
```

### Client Interface
```python
from tools import create_client

client = create_client()

# Structured access
portfolio = client.portfolio.optimize_portfolio(assets="AAPL,MSFT,GOOGL")
risk = client.risk.calculate_risk_metrics(assets="AAPL,MSFT,GOOGL")

# Conversational access
response = client.chat("Optimize a portfolio with tech stocks")
```

### Team Agent (for Agno Playground)
```python
from tools import FinanceTeam

team = FinanceTeam()
for response in team.run("What's the risk of holding AAPL and TSLA?"):
    print(response.content)
```

## Key Features

✅ **Individual Tools**: Each function calls specific APIs and returns structured data  
✅ **LLM-Friendly**: Tools can be called directly by language models  
✅ **Conversational**: Natural language interface through team agent  
✅ **Structured Access**: Organized client interface for programmatic use  
✅ **Error Handling**: Comprehensive error handling and validation  
✅ **Documentation**: Full docstrings and type hints for all tools  

All tools internally use the workflow components to make actual API calls to the financial analysis engines.
