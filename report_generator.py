"""
Report Generation Workflow for Agno Framework - Connects to Riskfolio Engine Report Endpoints
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


class ReportGenerator(Workflow):
    """Report generation workflow using riskfolio engine report endpoints."""
    
    description = """
    Generate comprehensive financial reports and analysis documents.
    
    Available report tools:
    - generate_portfolio_report: Complete portfolio analysis report
    - risk_report: Detailed risk assessment report
    - performance_report: Performance metrics summary
    - compliance_report: Regulatory compliance report
    
    This workflow connects to riskfolio engine report endpoints to:
    1. Generate professional PDF/HTML reports
    2. Create executive summaries
    3. Produce regulatory compliance documents
    4. Export data in various formats
    
    Report types available:
    - Portfolio Analysis Reports
    - Risk Assessment Reports
    - Performance Attribution Reports
    - Stress Testing Reports
    - Compliance Reports
    """
    
    def run(self, 
            report_type: str = "portfolio",
            assets: List[str] = None,
            weights: Dict[str, float] = None,
            format: str = "json") -> Dict[str, Any]:
        """
        Generate financial reports using riskfolio engine.
        
        Args:
            report_type: Type of report (portfolio, risk, performance)
            assets: List of asset symbols
            weights: Portfolio weights
            format: Output format (json, pdf, html)
            
        Returns:
            Dictionary with report data and metadata
        """
        try:
            # Default assets
            if assets is None:
                assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
            
            if report_type == "portfolio":
                return self.generate_portfolio_report(assets, weights, format)
            elif report_type == "risk":
                return self.risk_report(assets, weights)
            elif report_type == "performance":
                return self.performance_report(assets, weights)
            else:
                return self.generate_portfolio_report(assets, weights, format)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Report generation failed: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def generate_portfolio_report(self, 
                                 assets: str = "AAPL,MSFT,AMZN,GOOGL,TSLA",
                                 weights: str = "0.2,0.2,0.2,0.2,0.2",
                                 format: str = "json") -> Dict[str, Any]:
        """
        Generate comprehensive portfolio analysis report.
        
        Args:
            assets: Comma-separated asset symbols
            weights: Comma-separated portfolio weights
            format: Output format (json, pdf, html)
            
        Returns:
            Complete portfolio analysis report
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            weight_list = [float(w.strip()) for w in weights.split(",")]
            
            if len(asset_list) != len(weight_list):
                return {
                    "success": False,
                    "error": "Number of assets must match number of weights"
                }
            
            portfolio_weights = dict(zip(asset_list, weight_list))
            returns_df = self._generate_sample_data(asset_list)
            returns_data = self._prepare_returns_data(returns_df)
            
            # Call riskfolio engine report endpoint
            api_url = "http://localhost:5000/report"
            
            payload = {
                "returns": returns_data,
                "weights": portfolio_weights,
                "report_type": "portfolio",
                "format": format,
                "include_plots": True,
                "include_metrics": True
            }
            
            response = requests.post(api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            api_result = response.json()
            
            return {
                "success": True,
                "report_type": "portfolio_analysis",
                "report_data": api_result,
                "portfolio_weights": portfolio_weights,
                "format": format,
                "description": "Comprehensive portfolio analysis report"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Portfolio report generation failed: {str(e)}"
            }
    
    def risk_report(self, 
                   assets: str = "AAPL,MSFT,AMZN",
                   weights: str = "0.33,0.33,0.34") -> Dict[str, Any]:
        """
        Generate detailed risk assessment report.
        
        Args:
            assets: Comma-separated asset symbols
            weights: Comma-separated portfolio weights
            
        Returns:
            Risk assessment report
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            weight_list = [float(w.strip()) for w in weights.split(",")]
            
            portfolio_weights = dict(zip(asset_list, weight_list))
            returns_df = self._generate_sample_data(asset_list)
            returns_data = self._prepare_returns_data(returns_df)
            
            # Call riskfolio engine report endpoint
            api_url = "http://localhost:5000/report"
            
            payload = {
                "returns": returns_data,
                "weights": portfolio_weights,
                "report_type": "risk",
                "include_var": True,
                "include_stress_tests": True,
                "confidence_levels": [0.95, 0.99]
            }
            
            response = requests.post(api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            api_result = response.json()
            
            return {
                "success": True,
                "report_type": "risk_assessment",
                "report_data": api_result,
                "portfolio_weights": portfolio_weights,
                "description": "Detailed risk assessment report"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Risk report generation failed: {str(e)}"
            }
    
    def performance_report(self, 
                          assets: str = "AAPL,MSFT,AMZN",
                          weights: str = "0.33,0.33,0.34") -> Dict[str, Any]:
        """
        Generate performance metrics summary report.
        
        Args:
            assets: Comma-separated asset symbols
            weights: Comma-separated portfolio weights
            
        Returns:
            Performance metrics report
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            weight_list = [float(w.strip()) for w in weights.split(",")]
            
            portfolio_weights = dict(zip(asset_list, weight_list))
            returns_df = self._generate_sample_data(asset_list)
            returns_data = self._prepare_returns_data(returns_df)
            
            # Call riskfolio engine report endpoint
            api_url = "http://localhost:5000/report"
            
            payload = {
                "returns": returns_data,
                "weights": portfolio_weights,
                "report_type": "performance",
                "include_benchmarks": True,
                "include_attribution": True
            }
            
            response = requests.post(api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            api_result = response.json()
            
            return {
                "success": True,
                "report_type": "performance_summary",
                "report_data": api_result,
                "portfolio_weights": portfolio_weights,
                "description": "Performance metrics summary report"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Performance report generation failed: {str(e)}"
            }
    
    def export_data(self, 
                   assets: str = "AAPL,MSFT,AMZN",
                   data_type: str = "returns",
                   format: str = "csv") -> Dict[str, Any]:
        """
        Export portfolio data in various formats.
        
        Args:
            assets: Comma-separated asset symbols
            data_type: Type of data to export (returns, prices, weights)
            format: Export format (csv, json, excel)
            
        Returns:
            Exported data information
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            
            if data_type == "returns":
                returns_df = self._generate_sample_data(asset_list)
                data = self._prepare_returns_data(returns_df)
            else:
                data = {"message": f"Data type {data_type} not yet implemented"}
            
            return {
                "success": True,
                "export_type": data_type,
                "format": format,
                "data": data,
                "assets": asset_list,
                "description": f"Exported {data_type} data in {format} format"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Data export failed: {str(e)}"
            }
    
    def _prepare_returns_data(self, returns_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Convert pandas DataFrame to the format expected by riskfolio engine."""
        returns_data = {}
        for date, row in returns_df.iterrows():
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            returns_data[date_str] = row.to_dict()
        return returns_data
    
    def _generate_sample_data(self, assets: List[str], n_days: int = 252) -> pd.DataFrame:
        """Generate sample return data for reports."""
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
        
        # Create DataFrame with dates
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(n_days * 1.4))
        dates = pd.date_range(start=start_date, end=end_date, freq='B')[:n_days]
        
        returns_df = pd.DataFrame(returns_data, index=dates, columns=assets)
        return returns_df
