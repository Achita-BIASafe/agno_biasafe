"""
Base client for making API requests to the Riskfolio Engine.
"""

import os
from dotenv import load_dotenv
import requests
import json
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Any

# Load environment variables
load_dotenv()

# Set up basic logging
logger = logging.getLogger("riskfolio_client")

class RiskfolioClient:
    """
    Client for interacting with the Riskfolio API endpoints.
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialize the Riskfolio client.
        
        Args:
            base_url: The base URL of the Riskfolio API. If None, uses environment variable or localhost:5000.
        """
        self.base_url = base_url or os.getenv("RISKFOLIO_API_URL", "http://localhost:5000")
        self.session = requests.Session()
        logger.info(f"Initialized Riskfolio client with base URL: {self.base_url}")
    
    def _make_request(self, method: str, endpoint: str, data: Any = None, params: Dict = None) -> Dict:
        """
        Make an HTTP request to the Riskfolio API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without the base URL)
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        try:
            logger.debug(f"Making {method} request to {url}")
            if method.lower() == "get":
                response = self.session.get(url, params=params, headers=headers, timeout=30)
            elif method.lower() == "post":
                response = self.session.post(url, json=data, params=params, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise
    
    def health_check(self) -> Dict:
        """
        Check if the Riskfolio API is available.
        
        Returns:
            Health check response
        """
        try:
            response = self.session.get(f"{self.base_url}/")
            return {"status": "ok" if response.status_code == 200 else "error"}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def convert_to_dataframe(self, data: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Convert a nested dictionary of returns to a DataFrame.
        
        Args:
            data: Dictionary with dates as keys and dictionaries of asset returns as values
            
        Returns:
            pandas DataFrame of returns
        """
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index = pd.to_datetime(df.index)
        return df
    
    def convert_from_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Convert a DataFrame to the nested dictionary format required by the API.
        
        Args:
            df: pandas DataFrame of returns
            
        Returns:
            Dictionary with dates as keys and dictionaries of asset returns as values
        """
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        df_copy.index = df_copy.index.strftime("%Y-%m-%d")
        result = {}
        for date, row in df_copy.iterrows():
            result[date] = row.to_dict()
        return result
