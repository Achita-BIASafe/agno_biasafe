# AgnoRiskfolio

AgnoRiskfolio is a Python client library for interacting with the Riskfolio Engine API. It provides a comprehensive set of tools for portfolio optimization, risk analysis, data processing, and visualization.

## Features

- Portfolio optimization with various models and risk measures
- Risk metrics calculation and analysis
- Data processing and parameter estimation
- Visualization of portfolio weights, returns, drawdowns, and efficient frontiers
- Generation of comprehensive portfolio reports

## Installation

```bash
# Install from the local directory
pip install -e .

# Or clone the repository and install
git clone <repository-url>
cd agno_biasafe
pip install -e .
```

## Quick Start

```python
import pandas as pd
from agno_biasafe import create_client

# Create an AgnoRiskfolio client
client = create_client()

# Load returns data
returns_df = pd.read_csv('returns_data.csv', index_col=0, parse_dates=True)

# Optimize portfolio
optimal_weights = client.portfolio.optimize_portfolio(
    returns=returns_df,
    model="Classic",
    rm="MV",
    obj="MinRisk"
)

# Calculate risk metrics
risk_metrics = client.risk.calculate_risk_metrics(
    returns=returns_df,
    weights=optimal_weights
)

# Plot portfolio allocation
pie_chart = client.plotting.plot_pie_chart(
    weights=optimal_weights,
    title="Optimal Portfolio Allocation"
)

# Generate portfolio report
report = client.reports.generate_report(
    returns=returns_df,
    weights=optimal_weights
)
```

## Components

- **PortfolioOptimizer**: Portfolio optimization with various models and constraints
- **RiskAnalyzer**: Risk metrics calculation and analysis
- **DataHandler**: Data processing and parameter estimation
- **Plotter**: Visualization of portfolio weights, returns, and efficient frontiers
- **ReportGenerator**: Generation of comprehensive portfolio reports

## Configuration

You can configure the client by modifying the `config.py` file:

```python
# Default configuration
RISKFOLIO_API_BASE_URL = "http://127.0.0.1:5000"
REQUEST_TIMEOUT = 60
DEFAULT_MODEL = "Classic"
DEFAULT_RISK_MEASURE = "MV"
DEFAULT_OBJECTIVE = "MinRisk"
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_L = 2
```

## Examples

See the `example.py` file for detailed examples of how to use the AgnoRiskfolio client.

## Testing

Run the unit tests with:

```bash
python -m unittest tests.py
```
