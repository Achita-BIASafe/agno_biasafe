"""
Plotting Tools Workflow for Agno Framework - Professional chart generation
"""

from agno.workflow import Workflow
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import sys
import os
import requests
from api_config import get_endpoint

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


class PlottingTools(Workflow):
    """Professional plotting and visualization workflow using riskfolio engine."""
    
    description = """
    Generate professional charts and visualizations for portfolio analysis.
    
    Available plotting tools:
    - plot_efficient_frontier: Efficient frontier visualization
    - plot_pie_chart: Portfolio allocation pie charts
    - plot_drawdown: Drawdown analysis charts
    - plot_returns: Returns distribution histograms
    
    This workflow connects to riskfolio engine plotting endpoints to:
    1. Generate interactive efficient frontier charts
    2. Create portfolio allocation visualizations
    3. Plot drawdown and risk analysis charts
    4. Visualize returns distributions and statistics
    
    Chart types available:
    - Efficient Frontier: Risk-return trade-off visualization
    - Pie Charts: Portfolio weight allocation
    - Drawdown: Portfolio drawdown over time
    - Returns: Distribution and histogram plots
    """
    
    def run(self, 
            chart_type: str = "efficient_frontier",
            assets: List[str] = None,
            weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate charts using riskfolio engine plotting endpoints.
        
        Args:
            chart_type: Type of chart (efficient_frontier, pie_chart, drawdown, returns)
            assets: List of asset symbols
            weights: Portfolio weights for visualization
            
        Returns:
            Dictionary with chart generation results
        """
        try:
            # Default assets
            if assets is None:
                assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
            
            if chart_type == "efficient_frontier":
                return self.plot_efficient_frontier(assets)
            elif chart_type == "pie_chart":
                return self.plot_pie_chart(assets, weights)
            elif chart_type == "drawdown":
                return self.plot_drawdown(assets, weights)
            elif chart_type == "returns":
                return self.plot_returns(assets)
            else:
                return self.plot_efficient_frontier(assets)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Chart generation failed: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def plot_efficient_frontier(self, assets: str = "AAPL,MSFT,AMZN,GOOGL,TSLA") -> Dict[str, Any]:
        """
        Generate efficient frontier chart using riskfolio engine.
        
        Args:
            assets: Comma-separated asset symbols
            
        Returns:
            Efficient frontier chart generation results
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            returns_df = self._generate_sample_data(asset_list)
            
            # Prepare data for plotting
            weights_data = self._calculate_frontier_weights(returns_df)
            
            # Call riskfolio engine plot frontier endpoint
            api_url = get_endpoint("plot_frontier")
            
            payload = {
                "weights": weights_data,
                "asset_names": asset_list,
                "title": "Efficient Frontier",
                "width": 800,
                "height": 600
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            api_result = response.json()
            
            return {
                "success": True,
                "chart_type": "efficient_frontier",
                "chart_data": api_result,
                "assets": asset_list,
                "description": "Interactive efficient frontier visualization"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Efficient frontier plotting failed: {str(e)}"
            }
    
    def plot_pie_chart(self, 
                      assets: str = "AAPL,MSFT,AMZN,GOOGL",
                      weights: str = "0.25,0.25,0.25,0.25") -> Dict[str, Any]:
        """
        Generate portfolio allocation pie chart using riskfolio engine.
        
        Args:
            assets: Comma-separated asset symbols
            weights: Comma-separated portfolio weights
            
        Returns:
            Pie chart generation results
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
            
            # Call riskfolio engine plot pie endpoint
            api_url = get_endpoint("plot_pie")
            
            payload = {
                "weights": portfolio_weights,
                "title": "Portfolio Allocation",
                "width": 600,
                "height": 600
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            api_result = response.json()
            
            return {
                "success": True,
                "chart_type": "pie_chart",
                "chart_data": api_result,
                "portfolio_weights": portfolio_weights,
                "description": "Portfolio allocation pie chart"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Pie chart plotting failed: {str(e)}"
            }
    
    def plot_drawdown(self, 
                     assets: str = "AAPL,MSFT,AMZN",
                     weights: str = "0.33,0.33,0.34") -> Dict[str, Any]:
        """
        Generate portfolio drawdown chart using riskfolio engine.
        
        Args:
            assets: Comma-separated asset symbols
            weights: Comma-separated portfolio weights
            
        Returns:
            Drawdown chart generation results
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
            
            # Calculate portfolio returns and drawdown
            portfolio_returns = returns_df.dot(pd.Series(portfolio_weights))
            cumulative_returns = (1 + portfolio_returns).cumprod()
            drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1) * 100
            
            # Call riskfolio engine plot drawdown endpoint
            api_url = get_endpoint("plot_drawdown")
            
            payload = {
                "returns": self._prepare_returns_data(returns_df),
                "weights": portfolio_weights,
                "title": "Portfolio Drawdown Analysis",
                "width": 800,
                "height": 400
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            api_result = response.json()
            
            return {
                "success": True,
                "chart_type": "drawdown",
                "chart_data": api_result,
                "max_drawdown": drawdown.min(),
                "portfolio_weights": portfolio_weights,
                "description": "Portfolio drawdown analysis chart"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Drawdown plotting failed: {str(e)}"
            }
    
    def plot_returns(self, assets: str = "AAPL,MSFT,AMZN") -> Dict[str, Any]:
        """
        Generate returns distribution histogram using riskfolio engine.
        
        Args:
            assets: Comma-separated asset symbols
            
        Returns:
            Returns histogram generation results
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            returns_df = self._generate_sample_data(asset_list)
            
            # Call riskfolio engine plot histogram endpoint
            api_url = get_endpoint("plot_hist")
            
            payload = {
                "returns": self._prepare_returns_data(returns_df),
                "title": "Returns Distribution",
                "bins": 50,
                "width": 800,
                "height": 500
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            api_result = response.json()
            
            return {
                "success": True,
                "chart_type": "returns_histogram",
                "chart_data": api_result,
                "assets": asset_list,
                "description": "Returns distribution histogram"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Returns plotting failed: {str(e)}"
            }
    
    def _calculate_frontier_weights(self, returns_df: pd.DataFrame, points: int = 20) -> Dict[str, List[float]]:
        """Calculate sample efficient frontier weights for plotting."""
        n_assets = len(returns_df.columns)
        weights_data = {}
        
        # Generate sample weights for efficient frontier points
        for i in range(points):
            # Create random weights that sum to 1
            weights = np.random.dirichlet(np.ones(n_assets))
            weights_data[f"point_{i}"] = weights.tolist()
        
        return weights_data
    
    def _prepare_returns_data(self, returns_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Convert pandas DataFrame to the format expected by riskfolio engine."""
        returns_data = {}
        for date, row in returns_df.iterrows():
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            returns_data[date_str] = row.to_dict()
        return returns_data
    
    def _generate_sample_data(self, assets: List[str], n_days: int = 252) -> pd.DataFrame:
        """Generate sample return data for testing."""
        np.random.seed(42)
        n_assets = len(assets)
        
        # Generate realistic returns with correlations
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
