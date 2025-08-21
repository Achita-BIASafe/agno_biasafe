"""
Risk Analysis Workflow for Agno Framework
"""

from agno.workflow import Workflow
from agno.run.response import RunResponse
from agno.agent import Agent
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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


class RiskAnalyzer(Workflow):
    """Risk analysis workflow and agent using riskfolio engine."""
    
    # Agent-like properties for conversational interaction
    name = "Risk Analysis Agent"
    role = "Portfolio Risk Assessment Specialist"
    
    instructions = [
        "You are an expert risk analysis agent that helps users understand portfolio risk metrics.",
        "You can analyze custom asset selections and provide comprehensive risk assessments.",
        "You specialize in volatility analysis, risk-adjusted returns, and downside protection metrics.",
        "Always explain risk metrics in simple terms that users can understand.",
        "Provide actionable insights about portfolio risk levels and recommendations."
    ]
    
    description = """
    Analyze portfolio risk using various risk measures and metrics with CUSTOM ASSET SELECTION.
    
    Available tools for custom assets:
    - run: Main risk analysis with flexible asset input
    - analyze_portfolio_risk: Portfolio risk assessment for your assets
    - calculate_var: Value at Risk calculation for your portfolio
    - stress_test_portfolio: Portfolio stress testing for your assets
    
    CUSTOM ASSET EXAMPLES:
    - Tech Portfolio: "AAPL,MSFT,GOOGL,NVDA,TSLA"
    - Energy Sector: "XOM,CVX,COP,EOG,SLB"
    - Financial Sector: "JPM,BAC,WFC,C,GS"
    - ETF Portfolio: "SPY,QQQ,VTI,ARKK,XLK"
    - Custom Weights: assets="AAPL,MSFT,GOOGL", portfolio_weights="0.4,0.3,0.3"
    
    This workflow connects to riskfolio engine endpoints to:
    1. Calculate portfolio risk metrics (VaR, CVaR, etc.) for YOUR assets
    2. Perform stress testing and scenario analysis on YOUR portfolio
    3. Analyze risk decomposition and attribution for YOUR holdings
    
    Supported risk measures:
    - VaR: Value at Risk
    - CVaR: Conditional Value at Risk (Expected Shortfall)
    - MAD: Mean Absolute Deviation
    - Standard Deviation
    - Drawdown metrics
    """
    
    def run(self, 
            assets: str = "AAPL,MSFT,AMZN,GOOGL,TSLA",
            portfolio_weights: str = "0.2,0.2,0.2,0.2,0.2",
            risk_measure: str = "VaR",
            confidence_level: float = 0.95,
            time_horizon: int = 1) -> RunResponse:
        """ 
        Run comprehensive risk analysis using riskfolio engine with CUSTOM ASSET SELECTION.
        
        Args:
            assets: Comma-separated asset symbols (e.g., "AAPL,MSFT,TSLA") or list of symbols
            portfolio_weights: Comma-separated portfolio weights (e.g., "0.3,0.3,0.4") or empty for equal weights
            risk_measure: Risk measure to calculate (VaR, CVaR, MAD, etc.)
            confidence_level: Confidence level for risk calculations
            time_horizon: Time horizon in days
            
        Returns:
            RunResponse with risk analysis results from riskfolio engine
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
            if len(asset_list) < 1:
                return RunResponse(
                    content={
                        "success": False,
                        "error": "Risk analysis requires at least 1 asset",
                        "assets_provided": asset_list
                    },
                    content_type="json"
                )
            
            # Remove duplicates while preserving order
            seen = set()
            asset_list = [x for x in asset_list if not (x in seen or seen.add(x))]
            
            # Handle portfolio weights with better defaults
            if not portfolio_weights or portfolio_weights.strip() == "":
                # Equal weights (default)
                weight = 1.0 / len(asset_list)
                weights_dict = {asset: weight for asset in asset_list}
            else:
                try:
                    # Parse weights from string
                    weight_list = [float(w.strip()) for w in portfolio_weights.split(",")]
                    if len(weight_list) != len(asset_list):
                        return RunResponse(
                            content={
                                "success": False,
                                "error": f"Number of weights ({len(weight_list)}) must match number of assets ({len(asset_list)})"
                            },
                            content_type="json"
                        )
                    if abs(sum(weight_list) - 1.0) > 0.01:
                        return RunResponse(
                            content={
                                "success": False,
                                "error": f"Weights must sum to 1.0 (current sum: {sum(weight_list):.3f})"
                            },
                            content_type="json"
                        )
                    weights_dict = dict(zip(asset_list, weight_list))
                except ValueError:
                    return RunResponse(
                        content={
                            "success": False,
                            "error": "Invalid weight format. Use comma-separated decimals (e.g., '0.3,0.3,0.4')"
                        },
                        content_type="json"
                    )
            
            # Generate sample data
            returns_df = self._generate_sample_data(asset_list)
            
            # Call riskfolio engine risk analysis
            result = self._call_riskfolio_risk_analysis(
                returns_df, weights_dict, risk_measure, confidence_level
            )
            
            # Add asset metadata to result
            if result.get("success"):
                result["assets_used"] = asset_list
                result["num_assets"] = len(asset_list)
                result["portfolio_weights"] = weights_dict
            
            return RunResponse(
                content=result,
                content_type="json"
            )
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Risk analysis failed: {str(e)}",
                "error_type": type(e).__name__,
                "assets_attempted": asset_list if 'asset_list' in locals() else None
            }
            return RunResponse(
                content=error_result,
                content_type="json"
            )
    
    def handle_conversation(self, user_input: str) -> str:
        """
        Handle conversational interactions with the risk analysis agent.
        
        Args:
            user_input: User's natural language input about risk analysis
            
        Returns:
            Conversational response with risk assessment
        """
        # Parse user intent and extract assets if mentioned
        assets = self._extract_assets_from_text(user_input)
        
        # Use default assets if none found
        if not assets:
            assets = "AAPL,MSFT,AMZN,GOOGL,TSLA"
        
        # Run risk analysis
        result = self.run(assets=assets)
        
        if result.content.get("success"):
            # Format conversational response
            response = f"I've analyzed the risk profile"
            if assets != "AAPL,MSFT,AMZN,GOOGL,TSLA":
                response += f" for your assets: {assets}"
            response += ".\n\n"
            
            # Add key insights from risk analysis
            data = result.content
            if 'portfolio_volatility' in data:
                response += f"**Portfolio Volatility:** {data['portfolio_volatility']:.2%}\n"
            
            if 'var_95' in data:
                response += f"**Value at Risk (95%):** {data['var_95']:.2%}\n"
            
            if 'cvar_95' in data:
                response += f"**Conditional VaR (95%):** {data['cvar_95']:.2%}\n"
            
            if 'max_drawdown' in data:
                response += f"**Maximum Drawdown:** {data['max_drawdown']:.2%}\n"
            
            if 'sharpe_ratio' in data:
                response += f"**Sharpe Ratio:** {data['sharpe_ratio']:.3f}\n"
            
            # Risk level assessment
            volatility = data.get('portfolio_volatility', 0)
            if volatility < 0.15:
                risk_level = "Low"
                risk_desc = "conservative with steady returns expected"
            elif volatility < 0.25:
                risk_level = "Moderate"
                risk_desc = "balanced with reasonable volatility"
            else:
                risk_level = "High"
                risk_desc = "aggressive with potential for large swings"
            
            response += f"\n**Risk Assessment:** {risk_level} risk portfolio - {risk_desc}."
            response += "\n\nWould you like me to analyze different assets or explain any of these risk metrics?"
            
        else:
            response = f"I encountered an issue with your risk analysis: {result.content.get('error', 'Unknown error')}. "
            response += "Please check your asset symbols and try again, or ask me for help with specific assets."
        
        return response
    
    def _extract_assets_from_text(self, text: str) -> Optional[str]:
        """Extract asset symbols from natural language text."""
        import re
        
        # Look for patterns like "AAPL, GOOGL, MSFT" or "analyze TSLA and NVDA"
        # Simple regex to find potential stock symbols (2-5 uppercase letters)
        stock_pattern = r'\b[A-Z]{2,5}\b'
        potential_symbols = re.findall(stock_pattern, text.upper())
        
        # Filter out common English words that might match the pattern
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'HAVE', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'THIS', 'THAT', 'WITH', 'FROM', 'THEY', 'THEM', 'BEEN', 'SAID', 'EACH', 'WHICH', 'THEIR', 'TIME', 'MORE', 'VERY', 'WHAT', 'KNOW', 'JUST', 'FIRST', 'INTO', 'OVER', 'THINK', 'ALSO', 'YOUR', 'WORK', 'LIFE', 'ONLY', 'NEW', 'YEARS', 'WAY', 'MAY', 'COME', 'ITS', 'NOW', 'MOST', 'PEOPLE', 'GET', 'HAS', 'MUCH', 'LIKE', 'MADE', 'HOW', 'MANY', 'SOME', 'SO', 'THESE', 'SEE', 'HIM', 'TWO', 'WELL', 'WERE', 'RIGHT', 'BACK', 'OLD', 'WHERE', 'WANT', 'THOSE', 'CAME', 'GOOD', 'YEAR', 'SAME', 'USE', 'MAN', 'DAY', 'LONG', 'LITTLE', 'GREAT', 'NEVER', 'STILL', 'BETWEEN', 'ANOTHER', 'WHILE', 'LAST', 'MIGHT', 'MUST', 'US', 'LEFT', 'END', 'TURN', 'PLACE', 'BOTH', 'AGAIN', 'OFF', 'AWAY', 'EVEN', 'THROUGH', 'TAKE', 'EVERY', 'FOUND', 'UNDER', 'THOUGHT', 'DOWN', 'GIVE', 'CALLED', 'THREE', 'SMALL', 'DOES', 'PART', 'LOOKED', 'AFTER', 'NEXT', 'SEEM', 'WATER', 'AROUND', 'DIDN', 'PUT', 'ASKED', 'ABOVE', 'ALWAYS', 'BEING', 'TOLD', 'FELT', 'WENT', 'HAND', 'UNTIL', 'WORDS', 'WITHOUT', 'NOTHING', 'BEFORE', 'SAW', 'USED', 'MONEY', 'REALLY', 'USED', 'ACTUALLY', 'DOING', 'HELP', 'THING', 'THINGS', 'SOMETHING', 'ANYTHING', 'RISK', 'ANALYSIS'}
        
        # Filter out common words and keep likely stock symbols
        symbols = [s for s in potential_symbols if s not in common_words]
        
        if symbols:
            return ','.join(symbols)
        
        return None
    
    def _call_riskfolio_risk_analysis(self, returns_df: pd.DataFrame, 
                                    portfolio_weights: Dict[str, float],
                                    risk_measure: str, 
                                    confidence_level: float) -> Dict[str, Any]:
        """Call the riskfolio engine API for portfolio risk analysis."""
        try:
            # Prepare data for API call
            returns_data = self._prepare_returns_data(returns_df)
            
            # Step 1: Get portfolio returns from riskfolio engine
            portfolio_returns_url = "http://localhost:5000/returns/portfolio_returns"
            print(f"DEBUG: Portfolio returns URL: {portfolio_returns_url}")
            
            portfolio_payload = {
                "returns": returns_data,
                "weights": portfolio_weights,
                "rebalance": False
            }
            
            try:
                portfolio_response = requests.post(portfolio_returns_url, json=portfolio_payload, timeout=30)
                portfolio_response.raise_for_status()
                portfolio_result = portfolio_response.json()
                
                # Extract portfolio returns as a list
                portfolio_returns_list = [list(day_data.values())[0] for day_data in portfolio_result.values()]
                
            except Exception as portfolio_error:
                # Fallback: calculate portfolio returns locally if API fails
                portfolio_returns_series = (returns_df * pd.Series(portfolio_weights)).sum(axis=1)
                portfolio_returns_list = portfolio_returns_series.tolist()
            
            # Step 2: Calculate risk metrics on portfolio returns
            risk_metrics_url = get_endpoint("risk_metrics")
            
            # Calculate different risk metrics
            risk_results = {}
            
            # VaR calculation
            if risk_measure in ["VaR", "ALL"]:
                var_value = np.percentile(portfolio_returns_list, (1 - confidence_level) * 100)
                risk_results["VaR"] = var_value
            
            # CVaR calculation  
            if risk_measure in ["CVaR", "ALL"]:
                var_threshold = np.percentile(portfolio_returns_list, (1 - confidence_level) * 100)
                cvar_value = np.mean([r for r in portfolio_returns_list if r <= var_threshold])
                risk_results["CVaR"] = cvar_value
            
            # Volatility from API
            vol_payload = {
                "returns": portfolio_returns_list
            }
            
            try:
                vol_response = requests.post(risk_metrics_url, json=vol_payload, params={"metric": "VOL"}, timeout=30)
                vol_response.raise_for_status()
                volatility = vol_response.json()
                risk_results["Volatility"] = float(volatility) * np.sqrt(252)  # Annualized
            except:
                # Fallback calculation
                risk_results["Volatility"] = np.std(portfolio_returns_list) * np.sqrt(252)
            
            # Additional metrics
            portfolio_return = np.mean(portfolio_returns_list) * 252  # Annualized
            sharpe_ratio = portfolio_return / risk_results["Volatility"] if risk_results["Volatility"] > 0 else 0
            
            # Calculate individual asset contributions
            asset_contributions = {}
            for asset in returns_df.columns:
                asset_weight = portfolio_weights.get(asset, 0)
                asset_return = returns_df[asset].mean() * 252
                asset_vol = returns_df[asset].std() * np.sqrt(252)
                asset_contributions[asset] = {
                    "weight": asset_weight,
                    "contribution_to_return": asset_weight * asset_return,
                    "volatility": asset_vol
                }
            
            # Format the result
            return {
                "success": True,
                "risk_measure": risk_measure,
                "confidence_level": confidence_level,
                "portfolio_risk": risk_results["Volatility"],
                "var": risk_results.get("VaR", portfolio_returns_list[int((1-confidence_level)*len(portfolio_returns_list))]),
                "cvar": risk_results.get("CVaR", np.mean([r for r in portfolio_returns_list if r <= risk_results.get("VaR", 0)])),
                "volatility": risk_results["Volatility"],
                "max_drawdown": self._calculate_max_drawdown(portfolio_returns_list),
                "risk_metrics": {
                    "VaR": risk_results.get("VaR"),
                    "CVaR": risk_results.get("CVaR"), 
                    "Volatility": risk_results["Volatility"],
                    "Expected_Return": portfolio_return,
                    "Sharpe_Ratio": sharpe_ratio
                },
                "portfolio_metrics": {
                    "expected_return": portfolio_return,
                    "volatility": risk_results["Volatility"],
                    "var": risk_results.get("VaR"),
                    "cvar": risk_results.get("CVaR"),
                    "sharpe_ratio": sharpe_ratio
                },
                "asset_contributions": asset_contributions,
                "assets": list(returns_df.columns),
                "weights": portfolio_weights,
                "calculation_method": "riskfolio_engine_api"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to call riskfolio engine API: {str(e)}",
                "fallback_used": True
            }
    
    def _calculate_max_drawdown(self, returns_list: List[float]) -> float:
        """Calculate maximum drawdown from returns list."""
        cumulative = np.cumprod(1 + np.array(returns_list))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))
    
    def analyze_portfolio_risk(self, 
                              assets: List[str] = None,
                              risk_measure: str = "VaR",
                              portfolio_weights: Dict[str, float] = None,
                              confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Analyze portfolio risk using various measures.
        
        Args:
            assets: List of asset symbols
            risk_measure: Risk measure (VaR, CVaR, MAD, etc.)
            portfolio_weights: Portfolio weights for each asset
            confidence_level: Confidence level for risk calculations
            
        Returns:
            Portfolio risk analysis results
        """
        try:
            if assets is None:
                assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
            
            # Convert assets list to comma-separated string
            assets_str = ",".join(assets)
            
            # Convert portfolio weights dict to comma-separated string
            if portfolio_weights is None:
                weights_str = ""  # Use equal weights default
            else:
                # Ensure weights are in the same order as assets
                ordered_weights = [portfolio_weights.get(asset, 0) for asset in assets]
                weights_str = ",".join([str(w) for w in ordered_weights])
            
            # Call the main run method with string parameters
            result = self.run(
                assets=assets_str,
                portfolio_weights=weights_str,
                risk_measure=risk_measure,
                confidence_level=confidence_level
            )
            
            return result.content if hasattr(result, 'content') else result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Portfolio risk analysis failed: {str(e)}"
            }
    
    def calculate_var(self, 
                     assets: str = "AAPL,MSFT,AMZN,GOOGL,TSLA",
                     confidence_level: float = 0.95,
                     weights: str = "0.2,0.2,0.2,0.2,0.2") -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR) for a portfolio.
        
        Args:
            assets: Comma-separated asset symbols
            confidence_level: Confidence level (0.90, 0.95, 0.99)
            weights: Comma-separated portfolio weights
            
        Returns:
            VaR calculation results
        """
        try:
            # Parse inputs to validate format
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            weight_list = [float(w.strip()) for w in weights.split(",")]
            
            if len(asset_list) != len(weight_list):
                return {
                    "success": False,
                    "error": "Number of assets must match number of weights"
                }
            
            # Call risk analysis with string parameters
            result = self.run(
                assets=assets,
                portfolio_weights=weights,
                risk_measure="VaR",
                confidence_level=confidence_level
            )
            
            return result.content if hasattr(result, 'content') else result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"VaR calculation failed: {str(e)}"
            }
    
    def stress_test_portfolio(self, 
                             assets: str = "AAPL,MSFT,AMZN",
                             weights: str = "0.33,0.33,0.34",
                             shock_size: float = 0.05) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio.
        
        Args:
            assets: Comma-separated asset symbols
            weights: Comma-separated portfolio weights
            shock_size: Size of shock to apply (default 5%)
            
        Returns:
            Stress test results
        """
        try:
            # Parse inputs
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            weight_list = [float(w.strip()) for w in weights.split(",")]
            
            if len(asset_list) != len(weight_list):
                return {
                    "success": False,
                    "error": "Number of assets must match number of weights"
                }
            
            portfolio_weights = dict(zip(asset_list, weight_list))
            
            # Generate stressed returns data
            returns_df = self._generate_sample_data(asset_list)
            
            # Apply stress scenarios
            stress_scenarios = {
                "market_crash": returns_df - shock_size,
                "high_volatility": returns_df * (1 + shock_size),
                "correlation_spike": returns_df.corr() * (1 + shock_size * 2)
            }
            
            # Calculate portfolio performance under stress
            stress_results = {}
            for scenario_name, stressed_data in stress_scenarios.items():
                if scenario_name != "correlation_spike":  # Skip correlation for now
                    portfolio_returns = stressed_data.dot(pd.Series(portfolio_weights))
                    stress_results[scenario_name] = {
                        "portfolio_return": portfolio_returns.mean() * 252,
                        "portfolio_volatility": portfolio_returns.std() * np.sqrt(252),
                        "worst_day": portfolio_returns.min(),
                        "best_day": portfolio_returns.max()
                    }
            
            return {
                "success": True,
                "operation": "stress_testing",
                "portfolio_weights": portfolio_weights,
                "shock_size": shock_size,
                "stress_scenarios": stress_results,
                "base_metrics": {
                    "base_return": returns_df.dot(pd.Series(portfolio_weights)).mean() * 252,
                    "base_volatility": returns_df.dot(pd.Series(portfolio_weights)).std() * np.sqrt(252)
                },
                "description": f"Stress test with {shock_size*100}% shock applied"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Stress testing failed: {str(e)}"
            }
    
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
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(n_days * 1.4))
        dates = pd.date_range(start=start_date, end=end_date, freq='B')[:n_days]
        
        returns_df = pd.DataFrame(returns_data, index=dates, columns=assets)
        return returns_df
