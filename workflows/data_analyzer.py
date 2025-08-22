"""
Data Analysis Workflow for Agno Framework
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


class DataAnalyzer(Workflow):
    """Data analysis workflow and agent for financial market data analysis."""
    
    # Agent-like properties for conversational interaction
    name = "Data Analysis Agent"
    role = "Financial Data Analyst & Market Research Specialist"
    
    instructions = [
        "You are an expert financial data analysis agent that helps users understand market trends and patterns.",
        "You can analyze custom asset selections and provide comprehensive market data insights.",
        "You specialize in correlation analysis, return distributions, and historical performance metrics.",
        "Always provide clear explanations of statistical findings in accessible language.",
        "Offer actionable insights about market conditions and asset relationships."
    ]
    
    description = """
    Comprehensive financial data analysis workflow with CUSTOM ASSET SELECTION.
    
    Available tools for custom assets:
    - run: Main data analysis with flexible asset input
    - analyze_market_data: Simple market analysis for your assets
    - correlation_analysis: Asset correlation study for your portfolio
    - check_data_quality: Data quality assessment for your data
    
    CUSTOM ASSET EXAMPLES:
    - Market Analysis: "AAPL,MSFT,GOOGL,NVDA,TSLA"
    - Sector Analysis: "XLF,XLE,XLK,XLV,XLI"  (Financial, Energy, Tech, Healthcare, Industrial ETFs)
    - International: "VTI,VXUS,VWO,VEA,IEMG"
    - Bonds & Equities: "SPY,TLT,IEF,GLD,VNQ"
    - Crypto Analysis: "BTC-USD,ETH-USD,ADA-USD"
    
    This workflow:
    1. Analyzes market data patterns and trends for YOUR chosen assets
    2. Performs correlation and statistical analysis on YOUR portfolio
    3. Generates data quality reports for YOUR data
    4. Provides market insights and summaries for YOUR holdings
    
    Analysis types:
    - Market Overview: General market analysis for your assets
    - Correlation Analysis: Asset correlation matrices for your portfolio  
    - Trend Analysis: Price and volume trends for your assets
    - Quality Check: Data completeness and accuracy for your data
    
    Key features:
    - Statistical summaries for your chosen assets
    - Correlation matrices for your portfolio
    - Market regime detection for your holdings
    - Data visualization insights for your assets
    """
    
    def run(self, 
            analysis_type: str = "market_overview",
            assets: str = "AAPL,MSFT,AMZN,GOOGL,TSLA,SPY,QQQ",
            time_period: str = "1y",
            include_quality_check: bool = True) -> RunResponse:
        """
        Run data analysis with CUSTOM ASSET SELECTION.
        
        Args:
            analysis_type: Type of analysis to perform
            assets: Comma-separated asset symbols (e.g., "AAPL,MSFT,TSLA") or list of symbols
            time_period: Time period for analysis
            include_quality_check: Whether to include data quality checks
            
        Returns:
            RunResponse with analysis results
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
                asset_list = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'SPY', 'QQQ']
            
            # Validate assets
            if len(asset_list) < 1:
                return RunResponse(
                    content={
                        "success": False,
                        "error": "Data analysis requires at least 1 asset",
                        "assets_provided": asset_list
                    },
                    content_type="json"
                )
            
            # Remove duplicates while preserving order
            seen = set()
            asset_list = [x for x in asset_list if not (x in seen or seen.add(x))]
            
            # Generate sample data
            n_days = self._parse_period(time_period)
            returns_df = self._generate_sample_data(asset_list, n_days)
            prices_df = self._generate_price_data(returns_df)
            
            # Initialize client
            client = create_client()
            
            # Run analysis based on type
            if analysis_type == "correlation":
                result = self._correlation_analysis(client, returns_df, prices_df)
            elif analysis_type == "trend":
                result = self._trend_analysis(client, returns_df, prices_df)
            elif analysis_type == "quality_check":
                result = self._quality_check_analysis(client, returns_df, prices_df)
            else:
                result = self._market_overview_analysis(client, returns_df, prices_df)
            
            # Add quality check if requested
            if include_quality_check and analysis_type != "quality_check":
                quality_results = self._quality_check_analysis(client, returns_df, prices_df)
                result["data_quality"] = quality_results.get("quality_metrics", {})
            
            # Add asset metadata to result
            if result.get("success"):
                result["assets_used"] = asset_list
                result["num_assets"] = len(asset_list)
                result["time_period"] = time_period
                result["analysis_type"] = analysis_type
            
            return RunResponse(
                content=result,
                content_type="json"
            )
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Data analysis failed: {str(e)}",
                "error_type": type(e).__name__,
                "assets_attempted": asset_list if 'asset_list' in locals() else None
            }
            return RunResponse(
                content=error_result,
                content_type="json"
            )
    
    def handle_conversation(self, user_input: str) -> str:
        """
        Handle conversational interactions with the data analysis agent.
        
        Args:
            user_input: User's natural language input about data analysis
            
        Returns:
            Conversational response with data insights
        """
        # Parse user intent and extract assets if mentioned
        assets = self._extract_assets_from_text(user_input)
        
        # Determine analysis type from user input
        analysis_type = "overview"  # default
        if "correlation" in user_input.lower():
            analysis_type = "correlation"
        elif "quality" in user_input.lower() or "data quality" in user_input.lower():
            analysis_type = "quality"
        
        # Use default assets if none found
        if not assets:
            assets = "AAPL,MSFT,AMZN,GOOGL,TSLA"
        
        # Run data analysis
        result = self.run(assets=assets, analysis_type=analysis_type)
        
        if result.content.get("success"):
            # Format conversational response
            response = f"I've analyzed the market data"
            if assets != "AAPL,MSFT,AMZN,GOOGL,TSLA":
                response += f" for your assets: {assets}"
            response += f" with focus on {analysis_type} analysis.\n\n"
            
            # Add key insights based on analysis type
            data = result.content
            
            if analysis_type == "correlation" and 'correlation_matrix' in data:
                response += "**Correlation Analysis:**\n"
                correlations = data['correlation_matrix']
                # Find highest and lowest correlations
                highest_corr = max([max(row.values()) for row in correlations.values() if row])
                response += f"- Highest correlation: {highest_corr:.2%}\n"
                response += "- Assets with strong correlations tend to move together\n"
                
            elif analysis_type == "quality" and 'data_quality' in data:
                response += "**Data Quality Assessment:**\n"
                quality = data['data_quality']
                if 'completeness' in quality:
                    response += f"- Data completeness: {quality['completeness']:.1%}\n"
                if 'outliers_detected' in quality:
                    response += f"- Outliers detected: {quality['outliers_detected']}\n"
                    
            else:  # overview analysis
                if 'asset_metrics' in data:
                    response += "**Market Overview:**\n"
                    metrics = data['asset_metrics']
                    best_performer = max(metrics.keys(), key=lambda x: metrics[x].get('total_return', 0))
                    worst_performer = min(metrics.keys(), key=lambda x: metrics[x].get('total_return', 0))
                    
                    response += f"- Best performer: {best_performer} ({metrics[best_performer].get('total_return', 0):.2%})\n"
                    response += f"- Worst performer: {worst_performer} ({metrics[worst_performer].get('total_return', 0):.2%})\n"
                    
                    avg_volatility = np.mean([m.get('volatility', 0) for m in metrics.values()])
                    response += f"- Average volatility: {avg_volatility:.2%}\n"
            
            response += "\nWould you like me to analyze different assets or perform a different type of analysis?"
            
        else:
            response = f"I encountered an issue with your data analysis: {result.content.get('error', 'Unknown error')}. "
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
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'HAVE', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'THIS', 'THAT', 'WITH', 'FROM', 'THEY', 'THEM', 'BEEN', 'SAID', 'EACH', 'WHICH', 'THEIR', 'TIME', 'MORE', 'VERY', 'WHAT', 'KNOW', 'JUST', 'FIRST', 'INTO', 'OVER', 'THINK', 'ALSO', 'YOUR', 'WORK', 'LIFE', 'ONLY', 'NEW', 'YEARS', 'WAY', 'MAY', 'COME', 'ITS', 'NOW', 'MOST', 'PEOPLE', 'GET', 'HAS', 'MUCH', 'LIKE', 'MADE', 'HOW', 'MANY', 'SOME', 'SO', 'THESE', 'SEE', 'HIM', 'TWO', 'WELL', 'WERE', 'RIGHT', 'BACK', 'OLD', 'WHERE', 'WANT', 'THOSE', 'CAME', 'GOOD', 'YEAR', 'SAME', 'USE', 'MAN', 'DAY', 'LONG', 'LITTLE', 'GREAT', 'NEVER', 'STILL', 'BETWEEN', 'ANOTHER', 'WHILE', 'LAST', 'MIGHT', 'MUST', 'US', 'LEFT', 'END', 'TURN', 'PLACE', 'BOTH', 'AGAIN', 'OFF', 'AWAY', 'EVEN', 'THROUGH', 'TAKE', 'EVERY', 'FOUND', 'UNDER', 'THOUGHT', 'DOWN', 'GIVE', 'CALLED', 'THREE', 'SMALL', 'DOES', 'PART', 'LOOKED', 'AFTER', 'NEXT', 'SEEM', 'WATER', 'AROUND', 'DIDN', 'PUT', 'ASKED', 'ABOVE', 'ALWAYS', 'BEING', 'TOLD', 'FELT', 'WENT', 'HAND', 'UNTIL', 'WORDS', 'WITHOUT', 'NOTHING', 'BEFORE', 'SAW', 'USED', 'MONEY', 'REALLY', 'USED', 'ACTUALLY', 'DOING', 'HELP', 'THING', 'THINGS', 'SOMETHING', 'ANYTHING', 'DATA', 'ANALYSIS'}
        
        # Filter out common words and keep likely stock symbols
        symbols = [s for s in potential_symbols if s not in common_words]
        
        if symbols:
            return ','.join(symbols)
        
        return None
    
    def _market_overview_analysis(self, client, returns_df: pd.DataFrame, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive market overview analysis."""
        try:
            # Calculate basic statistics
            returns_stats = returns_df.describe()
            
            # Calculate additional metrics for each asset
            asset_metrics = {}
            for asset in returns_df.columns:
                asset_returns = returns_df[asset]
                asset_prices = prices_df[asset]
                
                # Performance metrics
                total_return = (asset_prices.iloc[-1] / asset_prices.iloc[0]) - 1
                annualized_return = (asset_prices.iloc[-1] / asset_prices.iloc[0]) ** (252 / len(asset_returns)) - 1
                volatility = asset_returns.std() * np.sqrt(252)
                sharpe_ratio = (annualized_return) / volatility if volatility > 0 else 0
                
                # Risk metrics
                max_drawdown = self._calculate_max_drawdown_from_prices(asset_prices)
                var_95 = np.percentile(asset_returns, 5)
                var_99 = np.percentile(asset_returns, 1)
                
                asset_metrics[asset] = {
                    "total_return": total_return,
                    "annualized_return": annualized_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "var_95": var_95,
                    "var_99": var_99,
                    "skewness": asset_returns.skew(),
                    "kurtosis": asset_returns.kurtosis()
                }
            
            # Market-wide metrics
            market_correlation = returns_df.corr().mean().mean()
            market_volatility = returns_df.mean(axis=1).std() * np.sqrt(252)
            
            # Identify market regimes (simplified)
            market_returns = returns_df.mean(axis=1)
            high_vol_periods = (market_returns.rolling(20).std() > market_returns.std() * 1.5).sum()
            
            return {
                "success": True,
                "analysis_type": "market_overview",
                "period": f"{returns_df.index[0].strftime('%Y-%m-%d')} to {returns_df.index[-1].strftime('%Y-%m-%d')}",
                "market_summary": {
                    "average_correlation": market_correlation,
                    "market_volatility": market_volatility,
                    "high_volatility_days": high_vol_periods,
                    "total_trading_days": len(returns_df)
                },
                "asset_metrics": asset_metrics,
                "statistical_summary": {
                    "mean_returns": returns_stats.loc['mean'].to_dict(),
                    "volatilities": returns_stats.loc['std'].to_dict(),
                    "min_returns": returns_stats.loc['min'].to_dict(),
                    "max_returns": returns_stats.loc['max'].to_dict()
                },
                "top_performers": {
                    "best_return": max(asset_metrics.keys(), key=lambda x: asset_metrics[x]["total_return"]),
                    "best_sharpe": max(asset_metrics.keys(), key=lambda x: asset_metrics[x]["sharpe_ratio"]),
                    "lowest_volatility": min(asset_metrics.keys(), key=lambda x: asset_metrics[x]["volatility"])
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Market overview analysis failed: {str(e)}"
            }
    
    def _correlation_analysis(self, client, returns_df: pd.DataFrame, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform detailed correlation analysis."""
        try:
            # Calculate correlation matrices
            correlation_matrix = returns_df.corr()
            
            # Rolling correlations (30-day window)
            rolling_corr = {}
            for i, asset1 in enumerate(returns_df.columns):
                for asset2 in returns_df.columns[i+1:]:
                    rolling_corr[f"{asset1}_{asset2}"] = returns_df[asset1].rolling(30).corr(returns_df[asset2])
            
            # Average correlations
            avg_correlations = {}
            for pair, corr_series in rolling_corr.items():
                avg_correlations[pair] = {
                    "mean": corr_series.mean(),
                    "std": corr_series.std(),
                    "min": corr_series.min(),
                    "max": corr_series.max()
                }
            
            # Correlation clustering
            correlation_levels = {
                "high_correlation": [],
                "medium_correlation": [],
                "low_correlation": []
            }
            
            for i, asset1 in enumerate(correlation_matrix.columns):
                for asset2 in correlation_matrix.columns[i+1:]:
                    corr_value = correlation_matrix.loc[asset1, asset2]
                    if abs(corr_value) > 0.7:
                        correlation_levels["high_correlation"].append((asset1, asset2, corr_value))
                    elif abs(corr_value) > 0.3:
                        correlation_levels["medium_correlation"].append((asset1, asset2, corr_value))
                    else:
                        correlation_levels["low_correlation"].append((asset1, asset2, corr_value))
            
            return {
                "success": True,
                "analysis_type": "correlation",
                "correlation_matrix": correlation_matrix.to_dict(),
                "correlation_summary": {
                    "average_correlation": correlation_matrix.mean().mean(),
                    "highest_correlation": correlation_matrix.max().max(),
                    "lowest_correlation": correlation_matrix.min().min()
                },
                "correlation_levels": correlation_levels,
                "rolling_correlation_stats": avg_correlations,
                "period": f"{returns_df.index[0].strftime('%Y-%m-%d')} to {returns_df.index[-1].strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Correlation analysis failed: {str(e)}"
            }
    
    def _trend_analysis(self, client, returns_df: pd.DataFrame, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform trend analysis."""
        try:
            trend_metrics = {}
            
            for asset in prices_df.columns:
                prices = prices_df[asset]
                returns = returns_df[asset]
                
                # Moving averages
                ma_20 = prices.rolling(20).mean()
                ma_50 = prices.rolling(50).mean()
                
                # Trend signals
                current_price = prices.iloc[-1]
                current_ma20 = ma_20.iloc[-1]
                current_ma50 = ma_50.iloc[-1]
                
                # Trend direction
                if current_price > current_ma20 > current_ma50:
                    trend = "strong_uptrend"
                elif current_price > current_ma20:
                    trend = "uptrend"
                elif current_price < current_ma20 < current_ma50:
                    trend = "strong_downtrend"
                elif current_price < current_ma20:
                    trend = "downtrend"
                else:
                    trend = "sideways"
                
                # Momentum indicators
                momentum_20 = (current_price / prices.iloc[-21]) - 1 if len(prices) > 21 else 0
                momentum_50 = (current_price / prices.iloc[-51]) - 1 if len(prices) > 51 else 0
                
                # Volatility trend
                vol_recent = returns.iloc[-20:].std()
                vol_historical = returns.std()
                vol_regime = "high" if vol_recent > vol_historical * 1.2 else "normal" if vol_recent > vol_historical * 0.8 else "low"
                
                trend_metrics[asset] = {
                    "trend_direction": trend,
                    "momentum_20d": momentum_20,
                    "momentum_50d": momentum_50,
                    "volatility_regime": vol_regime,
                    "current_vs_ma20": (current_price / current_ma20) - 1,
                    "current_vs_ma50": (current_price / current_ma50) - 1,
                    "price_range_position": (current_price - prices.min()) / (prices.max() - prices.min())
                }
            
            # Market-wide trends
            market_prices = prices_df.mean(axis=1)
            market_ma20 = market_prices.rolling(20).mean()
            market_trend = "bullish" if market_prices.iloc[-1] > market_ma20.iloc[-1] else "bearish"
            
            return {
                "success": True,
                "analysis_type": "trend",
                "individual_trends": trend_metrics,
                "market_trend": {
                    "overall_direction": market_trend,
                    "trending_up": sum(1 for m in trend_metrics.values() if "uptrend" in m["trend_direction"]),
                    "trending_down": sum(1 for m in trend_metrics.values() if "downtrend" in m["trend_direction"]),
                    "sideways": sum(1 for m in trend_metrics.values() if m["trend_direction"] == "sideways")
                },
                "period": f"{returns_df.index[0].strftime('%Y-%m-%d')} to {returns_df.index[-1].strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Trend analysis failed: {str(e)}"
            }
    
    def _quality_check_analysis(self, client, returns_df: pd.DataFrame, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform data quality check analysis."""
        try:
            quality_metrics = {}
            
            for asset in returns_df.columns:
                returns = returns_df[asset]
                prices = prices_df[asset]
                
                # Missing data
                missing_returns = returns.isna().sum()
                missing_prices = prices.isna().sum()
                
                # Outliers (using 3-sigma rule)
                returns_mean = returns.mean()
                returns_std = returns.std()
                outliers = ((returns - returns_mean).abs() > 3 * returns_std).sum()
                
                # Zero returns (potential data issues)
                zero_returns = (returns == 0).sum()
                
                # Extreme returns (beyond reasonable daily limits)
                extreme_positive = (returns > 0.2).sum()  # > 20% daily return
                extreme_negative = (returns < -0.2).sum()  # < -20% daily return
                
                # Data consistency checks
                negative_prices = (prices <= 0).sum()
                
                quality_metrics[asset] = {
                    "missing_data": {
                        "returns": missing_returns,
                        "prices": missing_prices,
                        "percentage": (missing_returns + missing_prices) / (2 * len(returns)) * 100
                    },
                    "outliers": {
                        "count": outliers,
                        "percentage": outliers / len(returns) * 100
                    },
                    "data_issues": {
                        "zero_returns": zero_returns,
                        "extreme_positive": extreme_positive,
                        "extreme_negative": extreme_negative,
                        "negative_prices": negative_prices
                    },
                    "data_quality_score": self._calculate_quality_score(
                        missing_returns + missing_prices, outliers, zero_returns,
                        extreme_positive + extreme_negative, negative_prices, len(returns)
                    )
                }
            
            # Overall data quality
            avg_quality_score = np.mean([metrics["data_quality_score"] for metrics in quality_metrics.values()])
            total_issues = sum(
                metrics["missing_data"]["returns"] + metrics["missing_data"]["prices"] +
                metrics["outliers"]["count"] + sum(metrics["data_issues"].values())
                for metrics in quality_metrics.values()
            )
            
            return {
                "success": True,
                "analysis_type": "quality_check",
                "quality_metrics": quality_metrics,
                "overall_quality": {
                    "average_quality_score": avg_quality_score,
                    "total_data_issues": total_issues,
                    "quality_grade": "A" if avg_quality_score > 0.9 else "B" if avg_quality_score > 0.8 else "C" if avg_quality_score > 0.7 else "D"
                },
                "period": f"{returns_df.index[0].strftime('%Y-%m-%d')} to {returns_df.index[-1].strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Quality check analysis failed: {str(e)}"
            }
    
    def _calculate_quality_score(self, missing: int, outliers: int, zeros: int, extremes: int, negatives: int, total: int) -> float:
        """Calculate a data quality score (0-1)."""
        total_issues = missing + outliers + zeros + extremes + negatives
        issue_rate = total_issues / total if total > 0 else 1
        return max(0, 1 - issue_rate)
    
    def _calculate_max_drawdown_from_prices(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series."""
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown.min()
    
    def _parse_period(self, period: str) -> int:
        """Parse period string to number of days."""
        period_map = {
            "1m": 21, "3m": 63, "6m": 126,
            "1y": 252, "2y": 504, "3y": 756
        }
        return period_map.get(period, 252)
    
    def _generate_sample_data(self, assets: List[str], n_days: int = 252) -> pd.DataFrame:
        """Generate sample return data."""
        np.random.seed(42)
        n_assets = len(assets)
        
        # Generate realistic returns
        mean_returns = np.random.uniform(0.0003, 0.0015, n_assets)
        volatilities = np.random.uniform(0.01, 0.04, n_assets)
        
        # Create correlation matrix
        correlations = np.random.uniform(0.1, 0.8, (n_assets, n_assets))
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
        
        # Add some data quality issues for testing
        # Random missing values
        missing_mask = np.random.random(returns_data.shape) < 0.002  # 0.2% missing
        returns_data[missing_mask] = np.nan
        
        # Occasional extreme values
        extreme_mask = np.random.random(returns_data.shape) < 0.001  # 0.1% extreme
        returns_data[extreme_mask] *= 5
        
        # Create DataFrame with dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(n_days * 1.4))
        dates = pd.date_range(start=start_date, end=end_date, freq='B')[:n_days]
        
        returns_df = pd.DataFrame(returns_data, index=dates, columns=assets)
        return returns_df
    
    def _generate_price_data(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Generate price data from returns."""
        initial_prices = np.random.uniform(50, 200, len(returns_df.columns))
        prices_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        
        for i, asset in enumerate(returns_df.columns):
            prices = [initial_prices[i]]
            for return_val in returns_df[asset]:
                if pd.isna(return_val):
                    prices.append(prices[-1])  # No change for missing returns
                else:
                    prices.append(prices[-1] * (1 + return_val))
            prices_df[asset] = prices[1:]  # Remove initial price
        
        return prices_df
    
    def analyze_market_data(self, assets: str = "AAPL,MSFT,AMZN") -> Dict[str, Any]:
        """
        Simple market data analysis tool.
        
        Args:
            assets: Comma-separated asset symbols
            
        Returns:
            Market analysis results including correlations and statistics
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            
            result = self.run(
                analysis_type="market_overview",
                assets=asset_list
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Market analysis failed: {str(e)}"
            }
    
    def correlation_analysis(self, assets: str = "AAPL,MSFT,AMZN,GOOGL") -> Dict[str, Any]:
        """
        Analyze correlations between assets.
        
        Args:
            assets: Comma-separated asset symbols
            
        Returns:
            Correlation analysis results
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            
            result = self.run(
                analysis_type="correlation",
                assets=asset_list
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Correlation analysis failed: {str(e)}"
            }
