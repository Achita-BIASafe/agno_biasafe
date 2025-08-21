"""
Portfolio Optimization Workflow for Agno Framework
"""

from agno.workflow import Workflow
from agno.agent import Agent
from agno.run.response import RunResponse
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json
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


class PortfolioOptimizer(Workflow):
    """Portfolio optimization workflow and agent using riskfolio engine."""
    
    # Agent-like properties for conversational interaction
    name = "Portfolio Optimizer Agent"
    role = "Expert Portfolio Manager & Risk Analyst"
    
    instructions = [
        "You are an expert portfolio optimization agent that helps users create optimal investment portfolios.",
        "You can analyze custom asset selections and provide portfolio recommendations.",
        "You use advanced risk measures and optimization techniques through the riskfolio engine.",
        "Always ask for clarification if the user's portfolio requirements are unclear.",
        "Provide clear explanations of your optimization results and recommendations."
    ]
    
    description = """
    Optimize portfolio weights using various risk measures and objectives with CUSTOM ASSET SELECTION.
    
    Available tools for custom assets:
    - optimize_portfolio: Optimize any assets (comma-separated, e.g., "TSLA,NVDA,AMD")
    - optimize_custom_portfolio: Individual asset inputs (up to 5 assets)
    - run: Main optimization with flexible asset input
    - calculate_efficient_frontier: Generate efficient frontier for your assets
    - calculate_sharpe_ratio: Calculate Sharpe ratios for your assets
    
    CUSTOM ASSET EXAMPLES:
    - Tech stocks: "AAPL,MSFT,GOOGL,NVDA,TSLA"
    - Blue chips: "JPM,JNJ,PG,KO,WMT" 
    - Growth stocks: "AMZN,NFLX,ROKU,ZOOM,SQ"
    - ETFs: "SPY,QQQ,VTI,ARKK,XLK"
    - Crypto: "BTC-USD,ETH-USD,ADA-USD"
    
    This workflow connects to riskfolio engine endpoints to:
    1. Process YOUR chosen assets and their return data
    2. Run portfolio optimization using different strategies
    3. Return optimal portfolio weights and metrics for YOUR assets
    
    Available optimization strategies:
    - MinRisk: Minimize portfolio risk (conservative)
    - MaxRet: Maximize expected return (aggressive)
    - Sharpe: Maximize risk-adjusted returns (balanced)
    - Conservative: CVaR-based risk minimization
    - Aggressive: Maximum return targeting
    
    Risk measures supported:
    - MV: Mean Variance (standard approach)
    - CVaR: Conditional Value at Risk (tail risk focus)
    - MAD: Mean Absolute Deviation (robust)
    - VaR: Value at Risk (downside protection)
    """
    
    def run(self, 
            assets: str = "AAPL,MSFT,AMZN,GOOGL,TSLA",
            model: str = "Classic",
            rm: str = "MV", 
            obj: str = "MinRisk",
            rf: float = 0.0,
            n_days: int = 252) -> RunResponse:
        """
        Run portfolio optimization using riskfolio engine with custom assets.
        
        Args:
            assets: Comma-separated asset symbols (e.g., "AAPL,MSFT,TSLA") or list of symbols
            model: Optimization model (Classic, BL, FM)
            rm: Risk measure (MV, CVaR, MAD, VaR)
            obj: Objective function (MinRisk, MaxRet, Sharpe, Utility)
            rf: Risk-free rate
            n_days: Number of days for sample data generation
            
        Returns:
            RunResponse with optimization results from riskfolio engine
        """
        try:
            # Handle different asset input formats
            if isinstance(assets, str):
                # If string passed, split by comma
                asset_list = [asset.strip().upper() for asset in assets.split(",")]
            elif isinstance(assets, list):
                # If list passed, clean and uppercase
                asset_list = [asset.strip().upper() for asset in assets]
            else:
                asset_list = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
            
            # Validate assets
            if len(asset_list) < 2:
                return RunResponse(
                    content={
                        "success": False,
                        "error": "Portfolio optimization requires at least 2 assets",
                        "assets_provided": asset_list
                    },
                    content_type="json"
                )
            
            # Remove duplicates while preserving order
            seen = set()
            asset_list = [x for x in asset_list if not (x in seen or seen.add(x))]
            
            # Generate sample data for the specified assets
            returns_df = self._generate_sample_data(asset_list, n_days)
            
            # Call riskfolio engine portfolio optimization endpoint
            result = self._call_riskfolio_optimization(returns_df, model, rm, obj, rf)
            
            # Add asset metadata to result
            if result.get("success"):
                result["assets_used"] = asset_list
                result["num_assets"] = len(asset_list)
                result["data_period_days"] = n_days
            
            return RunResponse(
                content=result,
                content_type="json"
            )
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Portfolio optimization failed: {str(e)}",
                "error_type": type(e).__name__,
                "assets_attempted": asset_list if 'asset_list' in locals() else None
            }
            return RunResponse(
                content=error_result,
                content_type="json"
            )
    
    def handle_conversation(self, user_input: str) -> str:
        """
        Handle conversational interactions with the portfolio optimization agent.
        
        Args:
            user_input: User's natural language input about portfolio optimization
            
        Returns:
            Conversational response with portfolio recommendations
        """
        # Parse user intent and extract assets if mentioned
        assets = self._extract_assets_from_text(user_input)
        
        # Use default assets if none found
        if not assets:
            assets = "AAPL,MSFT,AMZN,GOOGL,TSLA"
        
        # Run optimization
        result = self.run(assets=assets)
        
        if result.content.get("success"):
            # Format conversational response
            response = f"I've optimized your portfolio"
            if assets != "AAPL,MSFT,AMZN,GOOGL,TSLA":
                response += f" for the assets: {assets}"
            response += ".\n\n"
            
            # Add key insights from optimization
            data = result.content
            if 'weights' in data:
                response += "**Portfolio Allocation:**\n"
                for asset, weight in data['weights'].items():
                    response += f"- {asset}: {weight:.2%}\n"
            
            if 'expected_return' in data:
                response += f"\n**Expected Annual Return:** {data['expected_return']:.2%}"
            
            if 'volatility' in data:
                response += f"\n**Expected Volatility:** {data['volatility']:.2%}"
            
            if 'sharpe_ratio' in data:
                response += f"\n**Sharpe Ratio:** {data['sharpe_ratio']:.3f}"
            
            response += "\n\nWould you like me to analyze different assets or adjust the optimization parameters?"
            
        else:
            response = f"I encountered an issue with your portfolio optimization: {result.content.get('error', 'Unknown error')}. "
            response += "Please check your asset symbols and try again, or ask me for help with specific assets."
        
        return response
    
    def _extract_assets_from_text(self, text: str) -> Optional[str]:
        """Extract asset symbols from natural language text."""
        import re
        
        # Look for patterns like "AAPL, GOOGL, MSFT" or "optimize TSLA and NVDA"
        # Simple regex to find potential stock symbols (2-5 uppercase letters)
        stock_pattern = r'\b[A-Z]{2,5}\b'
        potential_symbols = re.findall(stock_pattern, text.upper())
        
        # Filter out common English words that might match the pattern
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'HAVE', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'THIS', 'THAT', 'WITH', 'FROM', 'THEY', 'THEM', 'BEEN', 'SAID', 'EACH', 'WHICH', 'THEIR', 'TIME', 'MORE', 'VERY', 'WHAT', 'KNOW', 'JUST', 'FIRST', 'INTO', 'OVER', 'THINK', 'ALSO', 'YOUR', 'WORK', 'LIFE', 'ONLY', 'NEW', 'YEARS', 'WAY', 'MAY', 'COME', 'ITS', 'NOW', 'MOST', 'PEOPLE', 'GET', 'HAS', 'MUCH', 'LIKE', 'MADE', 'HOW', 'MANY', 'SOME', 'SO', 'THESE', 'SEE', 'HIM', 'TWO', 'WELL', 'WERE', 'RIGHT', 'BACK', 'OLD', 'WHERE', 'WANT', 'THOSE', 'CAME', 'GOOD', 'YEAR', 'SAME', 'USE', 'MAN', 'DAY', 'LONG', 'LITTLE', 'GREAT', 'NEVER', 'STILL', 'BETWEEN', 'ANOTHER', 'WHILE', 'LAST', 'MIGHT', 'MUST', 'US', 'LEFT', 'END', 'TURN', 'PLACE', 'BOTH', 'AGAIN', 'OFF', 'AWAY', 'EVEN', 'THROUGH', 'TAKE', 'EVERY', 'FOUND', 'UNDER', 'THOUGHT', 'DOWN', 'GIVE', 'CALLED', 'THREE', 'SMALL', 'DOES', 'PART', 'LOOKED', 'AFTER', 'NEXT', 'SEEM', 'WATER', 'AROUND', 'DIDN', 'PUT', 'ASKED', 'ABOVE', 'ALWAYS', 'BEING', 'TOLD', 'FELT', 'WENT', 'HAND', 'UNTIL', 'WORDS', 'WITHOUT', 'NOTHING', 'BEFORE', 'SAW', 'USED', 'MONEY', 'REALLY', 'USED', 'ACTUALLY', 'DOING', 'HELP', 'THING', 'THINGS', 'SOMETHING', 'ANYTHING'}
        
        # Filter out common words and keep likely stock symbols
        symbols = [s for s in potential_symbols if s not in common_words]
        
        if symbols:
            return ','.join(symbols)
        
        return None
    
    def _call_riskfolio_optimization(self, returns_df: pd.DataFrame, model: str, rm: str, obj: str, rf: float) -> Dict[str, Any]:
        """Call the riskfolio engine portfolio optimization endpoint."""
        try:
            # Prepare data for API call
            returns_data = self._prepare_returns_data(returns_df)
            
            # Riskfolio engine portfolio endpoint
            api_url = get_endpoint("portfolio_optimize")
            
            payload = {
                "returns": returns_data,
                "model": model,
                "rm": rm,
                "obj": obj,
                "rf": rf,
                "l": 0,  # lower bound
                "u": 1,  # upper bound
                "w_max": 1.0,  # maximum weight per asset
                "n_sims": 1000  # number of simulations for some models
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            api_result = response.json()
            
            # Format the result
            return {
                "success": True,
                "method": f"{model} - {rm} - {obj}",
                "weights": api_result.get("weights", {}),
                "expected_return": api_result.get("expected_return"),
                "volatility": api_result.get("volatility"),
                "sharpe_ratio": api_result.get("sharpe_ratio"),
                "optimization_status": api_result.get("status", "completed"),
                "assets": list(returns_df.columns),
                "risk_measure": rm,
                "objective": obj,
                "model": model,
                "raw_response": api_result
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Failed to call riskfolio engine: {str(e)}",
                "fallback_used": False
            }
        except Exception as e:
            return {
                "success": False, 
                "error": f"Optimization processing failed: {str(e)}"
            }
    
    def _prepare_returns_data(self, returns_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Convert pandas DataFrame to the format expected by riskfolio engine."""
        returns_data = {}
        for date, row in returns_df.iterrows():
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            returns_data[date_str] = row.to_dict()
        return returns_data
    
    def calculate_efficient_frontier(self, assets: List[str] = None, rm: str = "MV", points: int = 50) -> Dict[str, Any]:
        """Calculate efficient frontier using riskfolio engine."""
        try:
            if assets is None:
                assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
            
            returns_df = self._generate_sample_data(assets)
            returns_data = self._prepare_returns_data(returns_df)
            
            api_url = get_endpoint("efficient_frontier")
            payload = {
                "returns": returns_data,
                "rm": rm,
                "points": points
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            return {
                "success": True,
                "frontier_data": response.json(),
                "risk_measure": rm,
                "points": points
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Efficient frontier calculation failed: {str(e)}"
            }
    
    def calculate_sharpe_ratio(self, assets: List[str] = None, rf: float = 0.0) -> Dict[str, Any]:
        """Calculate Sharpe ratio using riskfolio engine."""
        try:
            if assets is None:
                assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
            
            returns_df = self._generate_sample_data(assets)
            returns_data = self._prepare_returns_data(returns_df)
            
            api_url = get_endpoint("sharpe")
            payload = {
                "returns": returns_data,
                "rf": rf
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            return {
                "success": True,
                "sharpe_data": response.json(),
                "risk_free_rate": rf
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Sharpe ratio calculation failed: {str(e)}"
            }
    
    def optimize_custom_portfolio(self,
                                 asset1: str = "AAPL",
                                 asset2: str = "MSFT", 
                                 asset3: str = "GOOGL",
                                 asset4: str = "",
                                 asset5: str = "",
                                 strategy: str = "MinRisk") -> Dict[str, Any]:
        """
        Optimize portfolio with individual asset inputs (up to 5 assets).
        
        Args:
            asset1: First asset symbol (required)
            asset2: Second asset symbol (required) 
            asset3: Third asset symbol (required)
            asset4: Fourth asset symbol (optional)
            asset5: Fifth asset symbol (optional)
            strategy: Optimization strategy (MinRisk, MaxRet, Sharpe)
            
        Returns:
            Optimized portfolio for your custom asset selection
        """
        try:
            # Build asset list from individual inputs
            assets = [asset1.upper().strip(), asset2.upper().strip(), asset3.upper().strip()]
            
            # Add optional assets if provided
            if asset4.strip():
                assets.append(asset4.upper().strip())
            if asset5.strip():
                assets.append(asset5.upper().strip())
            
            # Remove empty strings and duplicates
            assets = list(dict.fromkeys([a for a in assets if a]))
            
            if len(assets) < 2:
                return {
                    "success": False,
                    "error": "Please provide at least 2 different asset symbols"
                }
            
            # Map strategy to technical parameters
            strategy_map = {
                "MinRisk": {"rm": "MV", "obj": "MinRisk"},
                "MaxRet": {"rm": "MV", "obj": "MaxRet"}, 
                "Sharpe": {"rm": "MV", "obj": "Sharpe"},
                "Conservative": {"rm": "CVaR", "obj": "MinRisk"},
                "Aggressive": {"rm": "MV", "obj": "MaxRet"}
            }
            
            params = strategy_map.get(strategy, {"rm": "MV", "obj": "MinRisk"})
            
            # Run optimization
            result = self.run(
                assets=assets,
                model="Classic",
                rm=params["rm"],
                obj=params["obj"]
            )
            
            # Enhance result with user-friendly info
            if result.get("success"):
                result["strategy_used"] = strategy
                result["portfolio_summary"] = {
                    "assets": assets,
                    "strategy": strategy,
                    "total_assets": len(assets)
                }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Custom portfolio optimization failed: {str(e)}",
                "assets_attempted": assets if 'assets' in locals() else []
            }
    
    def optimize_portfolio(self, 
                          assets: str = "AAPL,MSFT,AMZN,GOOGL,TSLA",
                          model: str = "Classic",
                          rm: str = "MV", 
                          obj: str = "MinRisk",
                          rf: float = 0.0,
                          custom_weights: str = None) -> Dict[str, Any]:
        """
        Portfolio optimization tool with flexible asset selection.
        
        Args:
            assets: Comma-separated list of asset symbols (e.g., "AAPL,MSFT,GOOGL")
            model: Optimization model (Classic, BL, FM)
            rm: Risk measure (MV, CVaR, MAD, VaR)
            obj: Objective function (MinRisk, MaxRet, Sharpe, Utility)
            rf: Risk-free rate (default: 0.0)
            custom_weights: Optional comma-separated initial weights (e.g., "0.3,0.3,0.4")
            
        Returns:
            Optimized portfolio weights and metrics for your specified assets
        """
        try:
            # Parse and validate assets
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            
            if len(asset_list) < 2:
                return {
                    "success": False,
                    "error": "Please provide at least 2 assets for portfolio optimization"
                }
            
            # Validate custom weights if provided
            if custom_weights:
                try:
                    weight_list = [float(w.strip()) for w in custom_weights.split(",")]
                    if len(weight_list) != len(asset_list):
                        return {
                            "success": False,
                            "error": f"Number of weights ({len(weight_list)}) must match number of assets ({len(asset_list)})"
                        }
                    if abs(sum(weight_list) - 1.0) > 0.01:
                        return {
                            "success": False,
                            "error": f"Weights must sum to 1.0 (current sum: {sum(weight_list):.3f})"
                        }
                except ValueError:
                    return {
                        "success": False,
                        "error": "Invalid weight format. Use comma-separated decimals (e.g., '0.3,0.3,0.4')"
                    }
            
            # Call the main optimization
            result = self.run(
                assets=asset_list,
                model=model,
                rm=rm,
                obj=obj,
                rf=rf
            )
            
            # Add asset information to result
            if result.get("success"):
                result["input_assets"] = asset_list
                result["num_assets"] = len(asset_list)
                if custom_weights:
                    result["initial_weights"] = dict(zip(asset_list, weight_list))
                
                # Add asset allocation summary
                if "weights" in result:
                    total_weight = sum(result["weights"].values()) if isinstance(result["weights"], dict) else 1.0
                    result["allocation_summary"] = {
                        "total_weight": total_weight,
                        "largest_position": max(result["weights"].values()) if isinstance(result["weights"], dict) else "N/A",
                        "most_weighted_asset": max(result["weights"], key=result["weights"].get) if isinstance(result["weights"], dict) else "N/A"
                    }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Portfolio optimization failed: {str(e)}",
                "assets_attempted": assets
            }
    
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
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(n_days * 1.4))
        dates = pd.date_range(start=start_date, end=end_date, freq='B')[:n_days]
        
        returns_df = pd.DataFrame(returns_data, index=dates, columns=assets)
        return returns_df
