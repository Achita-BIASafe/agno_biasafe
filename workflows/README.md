# Workflows Directory

This directory contains all workflow and agent classes for the Agno BIASafe AI financial analysis framework.

## Structure

### Workflows (extend `agno.workflow.Workflow`)
These are the core business logic components that handle specific financial analysis tasks:

- **`portfolio_optimizer.py`** - Portfolio optimization using various risk measures and objectives
- **`risk_analyzer.py`** - Risk analysis and measurement tools
- **`portfolio_backtester.py`** - Backtesting portfolio strategies and performance analysis
- **`data_analyzer.py`** - Data analysis and processing utilities
- **`plotting_tools.py`** - Visualization and plotting utilities for financial data
- **`report_generator.py`** - Generate comprehensive financial reports
- **`data_utilities.py`** - Helper functions and utilities for data processing

### Agents (extend `agno.agent.Agent`)
These provide conversational interfaces to the workflows:

- **`portfolio_optimizer_agent.py`** - Conversational interface for portfolio optimization
- **`risk_analysis_agent.py`** - Conversational interface for risk analysis
- **`data_analysis_agent.py`** - Conversational interface for data analysis
- **`portfolio_backtesting_agent.py`** - Conversational interface for backtesting

## Usage

### Import Workflows
```python
from workflows import PortfolioOptimizer, RiskAnalyzer
from workflows import PortfolioBacktester, DataAnalyzer

# Use workflow directly
optimizer = PortfolioOptimizer()
result = optimizer.run(assets="AAPL,MSFT,GOOGL", strategy="MinRisk")
```

### Import Agents
```python
from workflows import PortfolioOptimizerAgent, RiskAnalysisAgent

# Use agent for conversational interface
agent = PortfolioOptimizerAgent()
response = agent.handle_conversation("Optimize a tech portfolio with AAPL, MSFT, GOOGL")
```

## Integration with Tools

The tools in `/tools/` directory call these workflows to perform the actual financial analysis:

```python
# Example: tools/finance_tools.py calls workflows
from ..workflows import PortfolioOptimizer

def optimize_portfolio(assets: str):
    optimizer = PortfolioOptimizer()
    return optimizer.run(assets=assets)
```

## Dependencies

All workflows depend on:
- `agno.workflow.Workflow` base class
- `agno.agent.Agent` base class (for agents)
- `api_config.py` for endpoint configuration
- Various external libraries (pandas, numpy, requests, etc.)

## Configuration

Workflows connect to the riskfolio engine and other APIs using endpoints configured in `api_config.py`.
