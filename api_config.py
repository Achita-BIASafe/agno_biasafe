"""
API configuration for connecting to riskfolio-engine endpoints
Based on the actual endpoint definitions in riskfolio-engine/main.py
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base URL for riskfolio-engine
BASE_URL = os.getenv("RISKFOLIO_API_URL", "http://localhost:5000")

# API Endpoints - mapped to actual riskfolio-engine routes
ENDPOINTS = {
    # Portfolio endpoints (prefix: /portfolio)
    "portfolio_optimize": f"{BASE_URL}/portfolio/optimize",
    "portfolio_optimize_hc": f"{BASE_URL}/portfolio/optimize/hc",
    
    # Returns endpoints (prefix: /returns)  
    "portfolio_returns": f"{BASE_URL}/returns/portfolio_returns",
    
    # Efficient frontier endpoints (prefix: /efficient_frontier)
    "efficient_frontier": f"{BASE_URL}/efficient_frontier/weights",
    
    # Risk endpoints (prefix: /risk)
    "sharpe": f"{BASE_URL}/risk/sharpe",
    "alpha_risk": f"{BASE_URL}/risk/alpha",
    
    # Metrics endpoints (prefix: /metrics)
    "risk_metrics": f"{BASE_URL}/metrics/risk_metrics",
    "rolling_risk_metrics": f"{BASE_URL}/metrics/rolling_risk_metrics",
    
    # Plotting endpoints (prefix: /plotting)
    "plot_frontier": f"{BASE_URL}/plotting/plot_frontier",
    "plot_pie": f"{BASE_URL}/plotting/plot_pie",
    "plot_drawdown": f"{BASE_URL}/plotting/plot_drawdown",
    "plot_returns": f"{BASE_URL}/plotting/plot_returns",
    "plot_hist": f"{BASE_URL}/plotting/plot_hist",
    
    # Report endpoints (prefix: /report)
    "report_csv": f"{BASE_URL}/report/csv_report",
    "report_html": f"{BASE_URL}/report/html_report",
    
    # Parameters endpoints (prefix: /params)
    "covar": f"{BASE_URL}/params/covar",
    "mean_vector": f"{BASE_URL}/params/mean_vector",
    "cokurt": f"{BASE_URL}/params/cokurtosis",
    
    # Returns endpoints (prefix: /returns)
    "portfolio_returns": f"{BASE_URL}/returns/portfolio",
    
    # Statistics endpoints (prefix: /stats)
    "beta": f"{BASE_URL}/stats/beta",
    
    # Options endpoints (prefix: /options)
    "option_pricing": f"{BASE_URL}/options/price",
    
    # Constraints endpoints (prefix: /constraints)
    "asset_constraints": f"{BASE_URL}/constraints/assets",
    "asset_constraints_hrp": f"{BASE_URL}/constraints/assets/hrp",
    
    # Data endpoints (prefix: /data)
    "data": f"{BASE_URL}/data/fetch",
    
    # Health check
    "health": f"{BASE_URL}/health"
}

def get_endpoint(name: str) -> str:
    """
    Get the full URL for a named endpoint
    
    Args:
        name: Endpoint name from ENDPOINTS dict
        
    Returns:
        Full URL for the endpoint
        
    Raises:
        ValueError: If endpoint name is not found
    """
    if name not in ENDPOINTS:
        available_endpoints = ", ".join(ENDPOINTS.keys())
        raise ValueError(f"Unknown endpoint: {name}. Available endpoints: {available_endpoints}")
    return ENDPOINTS[name]

def set_base_url(url: str) -> None:
    """
    Update the base URL for all endpoints
    
    Args:
        url: New base URL (e.g., "http://localhost:5000" or "https://api.example.com")
    """
    global BASE_URL, ENDPOINTS
    BASE_URL = url.rstrip('/')  # Remove trailing slash
    
    # Update all endpoint URLs
    ENDPOINTS.update({
        # Portfolio endpoints
        "portfolio_optimize": f"{BASE_URL}/portfolio/optimize",
        "portfolio_optimize_hc": f"{BASE_URL}/portfolio/optimize/hc",
        
        # Efficient frontier endpoints
        "efficient_frontier": f"{BASE_URL}/efficient_frontier/weights",
        
        # Risk endpoints
        "sharpe": f"{BASE_URL}/risk/sharpe",
        "alpha_risk": f"{BASE_URL}/risk/alpha",
        
        # Metrics endpoints
        "risk_metrics": f"{BASE_URL}/metrics/risk_metrics",
        "rolling_risk_metrics": f"{BASE_URL}/metrics/rolling_risk_metrics",
        
        # Plotting endpoints
        "plot_frontier": f"{BASE_URL}/plotting/plot_frontier",
        "plot_pie": f"{BASE_URL}/plotting/plot_pie",
        "plot_drawdown": f"{BASE_URL}/plotting/plot_drawdown",
        "plot_returns": f"{BASE_URL}/plotting/plot_returns",
        "plot_hist": f"{BASE_URL}/plotting/plot_hist",
        
        # Report endpoints
        "report_csv": f"{BASE_URL}/report/csv_report",
        "report_html": f"{BASE_URL}/report/html_report",
        
        # Parameters endpoints
        "covar": f"{BASE_URL}/params/covar",
        "mean_vector": f"{BASE_URL}/params/mean_vector",
        "cokurt": f"{BASE_URL}/params/cokurtosis",
        
        # Returns endpoints
        "portfolio_returns": f"{BASE_URL}/returns/portfolio",
        
        # Statistics endpoints
        "beta": f"{BASE_URL}/stats/beta",
        
        # Options endpoints
        "option_pricing": f"{BASE_URL}/options/price",
        
        # Constraints endpoints
        "asset_constraints": f"{BASE_URL}/constraints/assets",
        "asset_constraints_hrp": f"{BASE_URL}/constraints/assets/hrp",
        
        # Data endpoints
        "data": f"{BASE_URL}/data/fetch",
        
        # Health check
        "health": f"{BASE_URL}/health"
    })

def list_endpoints() -> dict:
    """
    Get all available endpoints
    
    Returns:
        Dictionary of all endpoint names and URLs
    """
    return ENDPOINTS.copy()

def test_endpoint_connectivity() -> dict:
    """
    Test connectivity to the riskfolio-engine base URL
    
    Returns:
        Dictionary with connectivity test results
    """
    import requests
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        return {
            "success": True,
            "status_code": response.status_code,
            "base_url": BASE_URL,
            "message": "Successfully connected to riskfolio-engine"
        }
    except Exception as e:
        return {
            "success": False,
            "base_url": BASE_URL,
            "error": str(e),
            "message": "Failed to connect to riskfolio-engine"
        }
