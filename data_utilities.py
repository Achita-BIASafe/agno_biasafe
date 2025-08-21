"""
Data Utilities Workflow for Agno Framework - Advanced data handling and processing
"""

from agno.workflow import Workflow
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import sys
import os
import requests
import json

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from main import create_client
except ImportError:
    # Fallback for when running in agno context
    def create_client():
        from client import RiskfolioClient
        return RiskfolioClient()


class DataUtilities(Workflow):
    """Data utilities workflow for advanced data processing and transformation."""
    
    description = """
    Advanced data processing and utility functions for financial analysis.
    
    Available data tools:
    - calculate_covariance: Calculate covariance matrices
    - calculate_correlation: Calculate correlation matrices  
    - get_mean_vector: Calculate expected returns vector
    - process_returns_data: Clean and process returns data
    - validate_portfolio: Validate portfolio weights and constraints
    
    This workflow provides:
    1. Data validation and cleaning utilities
    2. Statistical calculations (covariance, correlation)
    3. Portfolio constraint validation
    4. Data format conversions
    5. Missing data handling
    
    Data processing features:
    - Covariance matrix calculations
    - Correlation analysis
    - Mean returns estimation
    - Data quality checks
    - Portfolio validation
    """
    
    def run(self, 
            operation: str = "covariance",
            assets: List[str] = None,
            weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Perform data utility operations using riskfolio engine.
        
        Args:
            operation: Type of operation (covariance, correlation, mean_vector)
            assets: List of asset symbols
            weights: Portfolio weights for validation
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Default assets
            if assets is None:
                assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
            
            if operation == "covariance":
                return self.calculate_covariance(assets)
            elif operation == "correlation":
                return self.calculate_correlation(assets)
            elif operation == "mean_vector":
                return self.get_mean_vector(assets)
            elif operation == "validate":
                return self.validate_portfolio(assets, weights)
            else:
                return self.calculate_covariance(assets)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Data operation failed: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def calculate_covariance(self, assets: str = "AAPL,MSFT,AMZN,GOOGL,TSLA") -> Dict[str, Any]:
        """
        Calculate covariance matrix using riskfolio engine.
        
        Args:
            assets: Comma-separated asset symbols
            
        Returns:
            Covariance matrix calculation results
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            returns_df = self._generate_sample_data(asset_list)
            returns_data = self._prepare_returns_data(returns_df)
            
            # Call riskfolio engine covar endpoint
            api_url = "http://localhost:5000/covar"
            
            payload = {
                "returns": returns_data,
                "method": "hist"  # Historical method
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            api_result = response.json()
            
            return {
                "success": True,
                "operation": "covariance_matrix",
                "covariance_data": api_result,
                "assets": asset_list,
                "method": "historical",
                "description": "Asset covariance matrix calculation"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Covariance calculation failed: {str(e)}"
            }
    
    def calculate_correlation(self, assets: str = "AAPL,MSFT,AMZN,GOOGL") -> Dict[str, Any]:
        """
        Calculate correlation matrix.
        
        Args:
            assets: Comma-separated asset symbols
            
        Returns:
            Correlation matrix results
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            returns_df = self._generate_sample_data(asset_list)
            
            # Calculate correlation matrix locally
            correlation_matrix = returns_df.corr()
            
            return {
                "success": True,
                "operation": "correlation_matrix",
                "correlation_data": correlation_matrix.to_dict(),
                "assets": asset_list,
                "average_correlation": correlation_matrix.mean().mean(),
                "description": "Asset correlation matrix calculation"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Correlation calculation failed: {str(e)}"
            }
    
    def get_mean_vector(self, assets: str = "AAPL,MSFT,AMZN") -> Dict[str, Any]:
        """
        Calculate expected returns vector using riskfolio engine.
        
        Args:
            assets: Comma-separated asset symbols
            
        Returns:
            Mean returns vector calculation results
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            returns_df = self._generate_sample_data(asset_list)
            returns_data = self._prepare_returns_data(returns_df)
            
            # Call riskfolio engine mean_vector endpoint
            api_url = "http://localhost:5000/mean_vector"
            
            payload = {
                "returns": returns_data,
                "method": "hist"  # Historical method
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            api_result = response.json()
            
            return {
                "success": True,
                "operation": "mean_returns_vector",
                "mean_vector_data": api_result,
                "assets": asset_list,
                "method": "historical",
                "description": "Expected returns vector calculation"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Mean vector calculation failed: {str(e)}"
            }
    
    def validate_portfolio(self, 
                          assets: str = "AAPL,MSFT,AMZN",
                          weights: str = "0.33,0.33,0.34") -> Dict[str, Any]:
        """
        Validate portfolio weights and constraints.
        
        Args:
            assets: Comma-separated asset symbols
            weights: Comma-separated portfolio weights
            
        Returns:
            Portfolio validation results
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            weight_list = [float(w.strip()) for w in weights.split(",")]
            
            if len(asset_list) != len(weight_list):
                return {
                    "success": False,
                    "error": "Number of assets must match number of weights",
                    "validation_failed": True
                }
            
            # Validation checks
            portfolio_weights = dict(zip(asset_list, weight_list))
            total_weight = sum(weight_list)
            
            validation_results = {
                "weights_sum_to_one": abs(total_weight - 1.0) < 0.001,
                "no_negative_weights": all(w >= 0 for w in weight_list),
                "no_weights_exceed_one": all(w <= 1.0 for w in weight_list),
                "total_weight": total_weight,
                "weight_distribution": {
                    "min_weight": min(weight_list),
                    "max_weight": max(weight_list),
                    "concentration": max(weight_list)  # Largest single position
                }
            }
            
            validation_results["is_valid"] = all([
                validation_results["weights_sum_to_one"],
                validation_results["no_negative_weights"],
                validation_results["no_weights_exceed_one"]
            ])
            
            return {
                "success": True,
                "operation": "portfolio_validation",
                "portfolio_weights": portfolio_weights,
                "validation_results": validation_results,
                "description": "Portfolio weights validation"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Portfolio validation failed: {str(e)}"
            }
    
    def process_returns_data(self, 
                            assets: str = "AAPL,MSFT,AMZN",
                            clean_method: str = "dropna") -> Dict[str, Any]:
        """
        Process and clean returns data.
        
        Args:
            assets: Comma-separated asset symbols
            clean_method: Data cleaning method (dropna, fillna, interpolate)
            
        Returns:
            Processed data summary
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            returns_df = self._generate_sample_data(asset_list)
            
            # Data quality assessment
            original_shape = returns_df.shape
            missing_count = returns_df.isnull().sum().sum()
            
            # Apply cleaning method
            if clean_method == "dropna":
                cleaned_df = returns_df.dropna()
            elif clean_method == "fillna":
                cleaned_df = returns_df.fillna(0)
            elif clean_method == "interpolate":
                cleaned_df = returns_df.interpolate()
            else:
                cleaned_df = returns_df.dropna()
            
            processed_shape = cleaned_df.shape
            
            return {
                "success": True,
                "operation": "data_processing",
                "original_shape": original_shape,
                "processed_shape": processed_shape,
                "missing_values_removed": missing_count,
                "cleaning_method": clean_method,
                "data_quality": {
                    "completeness": (1 - missing_count / (original_shape[0] * original_shape[1])) * 100,
                    "rows_retained": processed_shape[0] / original_shape[0] * 100
                },
                "description": f"Data processed using {clean_method} method"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Data processing failed: {str(e)}"
            }
    
    def _prepare_returns_data(self, returns_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Convert pandas DataFrame to the format expected by riskfolio engine."""
        returns_data = {}
        for date, row in returns_df.iterrows():
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            returns_data[date_str] = row.to_dict()
        return returns_data
    
    def _generate_sample_data(self, assets: List[str], n_days: int = 252) -> pd.DataFrame:
        """Generate sample return data with some missing values for testing."""
        np.random.seed(42)
        n_assets = len(assets)
        
        # Generate realistic returns
        mean_returns = np.random.uniform(0.0005, 0.0015, n_assets)
        volatilities = np.random.uniform(0.015, 0.035, n_assets)
        
        # Create correlation matrix
        correlations = np.random.uniform(0.3, 0.8, (n_assets, n_assets))
        correlations = (correlations + correlations.T) / 2
        np.fill_diagonal(correlations, 1.0)
        
        # Convert to covariance matrix
        covariance_matrix = np.outer(volatilities, volatilities) * correlations
        
        # Generate returns
        returns_data = np.random.multivariate_normal(
            mean=mean_returns,
            cov=covariance_matrix,
            size=n_days
        )
        
        # Add some missing values for testing
        missing_mask = np.random.random(returns_data.shape) < 0.02  # 2% missing
        returns_data[missing_mask] = np.nan
        
        # Create DataFrame with dates
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(n_days * 1.4))
        dates = pd.date_range(start=start_date, end=end_date, freq='B')[:n_days]
        
        returns_df = pd.DataFrame(returns_data, index=dates, columns=assets)
        return returns_df
