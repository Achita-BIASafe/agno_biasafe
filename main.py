"""
Main module for the agno_biasafe package. Provides easy access to the Riskfolio Engine API.
"""

from .client import RiskfolioClient

def create_client(base_url=None):
    """
    Create a new RiskfolioClient instance.
    
    Args:
        base_url: Base URL for the Riskfolio Engine API. If None, uses default.
        
    Returns:
        RiskfolioClient instance
    """
    return RiskfolioClient(base_url)
