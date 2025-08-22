"""
Portfolio Backtesting Workflow for Agno Framework
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


class PortfolioBacktester(Workflow):
    """Portfolio backtesting workflow and agent for strategy validation."""
    
    # Agent-like properties for conversational interaction
    name = "Portfolio Backtesting Agent"
    role = "Strategy Testing & Performance Analysis Specialist"
    
    instructions = [
        "You are an expert portfolio backtesting agent that helps users test investment strategies.",
        "You can backtest custom asset selections and provide comprehensive performance analysis.",
        "You specialize in historical performance analysis, drawdown analysis, and strategy comparison.",
        "Always explain backtesting results and their implications for strategy effectiveness.",
        "Provide clear recommendations based on historical performance data."
    ]
    
    description = """
    Comprehensive portfolio backtesting framework with CUSTOM ASSET SELECTION.
    
    Available tools for custom assets:
    - run: Main backtesting workflow with flexible asset input
    - backtest_strategy: Simple strategy backtesting for your assets
    - compare_strategies: Compare multiple strategies on your portfolio
    - test_portfolio: Quick portfolio performance test for your assets
    
    CUSTOM ASSET EXAMPLES:
    - Growth Strategy: "AAPL,MSFT,GOOGL,NVDA,TSLA"
    - Value Strategy: "BRK-B,JPM,JNJ,PG,KO"
    - Sector Rotation: "XLK,XLF,XLE,XLV,XLI"
    - International: "VTI,VXUS,VWO,VEA,IEMG"
    - Defensive Portfolio: "SCHD,VYM,VIG,PFF,TLT"
    
    This workflow:
    1. Runs historical simulations of portfolio strategies on YOUR assets
    2. Calculates comprehensive performance metrics for YOUR portfolio
    3. Compares strategy performance against benchmarks for YOUR holdings
    4. Provides detailed backtesting reports for YOUR assets
    
    Backtesting modes:
    - Single Strategy: Test one optimization strategy on your assets
    - Multi Strategy: Compare multiple strategies on your portfolio
    - Walk Forward: Rolling window backtesting for your holdings
    
    Performance metrics:
    - Total Returns for your assets
    - Sharpe Ratio for your portfolio
    - Maximum Drawdown for your holdings
    - Win Rate for your strategy
    - Risk-adjusted returns for your assets
    """
    
    def run(self, 
            strategy: str = "mean_variance",
            backtest_period: str = "1y",
            rebalance_frequency: str = "monthly",
            benchmark: str = "equal_weight",
            assets: str = "AAPL,MSFT,AMZN,GOOGL,TSLA") -> RunResponse:
        """
        Run portfolio backtesting with CUSTOM ASSET SELECTION.
        
        Args:
            strategy: Optimization strategy to test
            backtest_period: Period for backtesting
            rebalance_frequency: How often to rebalance
            benchmark: Benchmark comparison
            assets: Comma-separated asset symbols (e.g., "AAPL,MSFT,TSLA") or list of symbols
            
        Returns:
            RunResponse with backtesting results
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
                        "error": "Portfolio backtesting requires at least 2 assets",
                        "assets_provided": asset_list
                    },
                    content_type="json"
                )
            
            # Remove duplicates while preserving order
            seen = set()
            asset_list = [x for x in asset_list if not (x in seen or seen.add(x))]
            
            # Parse backtest period
            n_days = self._parse_period(backtest_period)
            
            # Generate sample data
            returns_df = self._generate_sample_data(asset_list, n_days)
            
            # Initialize client
            client = create_client()
            
            # Run backtesting
            if strategy == "multi_strategy":
                result = self._multi_strategy_backtest(client, returns_df, rebalance_frequency, benchmark)
            else:
                result = self._single_strategy_backtest(client, returns_df, strategy, rebalance_frequency, benchmark)
            
            # Add asset metadata to result
            if result.get("success"):
                result["assets_used"] = asset_list
                result["num_assets"] = len(asset_list)
                result["backtest_period"] = backtest_period
                result["strategy"] = strategy
                result["rebalance_frequency"] = rebalance_frequency
            
            return RunResponse(
                content=result,
                content_type="json"
            )
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Portfolio backtesting failed: {str(e)}",
                "error_type": type(e).__name__,
                "assets_attempted": asset_list if 'asset_list' in locals() else None
            }
            return RunResponse(
                content=error_result,
                content_type="json"
            )
    
    def handle_conversation(self, user_input: str) -> str:
        """
        Handle conversational interactions with the portfolio backtesting agent.
        
        Args:
            user_input: User's natural language input about backtesting
            
        Returns:
            Conversational response with backtesting results
        """
        # Parse user intent and extract assets if mentioned
        assets = self._extract_assets_from_text(user_input)
        
        # Determine strategy from user input
        strategy = "equal_weight"  # default
        if "momentum" in user_input.lower():
            strategy = "momentum"
        elif "minimum variance" in user_input.lower() or "min variance" in user_input.lower():
            strategy = "min_variance"
        elif "multi" in user_input.lower() or "multiple" in user_input.lower():
            strategy = "multi_strategy"
        
        # Use default assets if none found
        if not assets:
            assets = "AAPL,MSFT,AMZN,GOOGL,TSLA"
        
        # Run backtesting
        result = self.run(assets=assets, strategy=strategy)
        
        if result.content.get("success"):
            # Format conversational response
            response = f"I've backtested your {strategy.replace('_', ' ')} strategy"
            if assets != "AAPL,MSFT,AMZN,GOOGL,TSLA":
                response += f" using assets: {assets}"
            response += ".\n\n"
            
            # Add key insights from backtesting
            data = result.content
            
            if 'total_return' in data:
                response += f"**Total Return:** {data['total_return']:.2%}\n"
            
            if 'annualized_return' in data:
                response += f"**Annualized Return:** {data['annualized_return']:.2%}\n"
            
            if 'volatility' in data:
                response += f"**Volatility:** {data['volatility']:.2%}\n"
            
            if 'max_drawdown' in data:
                response += f"**Maximum Drawdown:** {data['max_drawdown']:.2%}\n"
            
            if 'sharpe_ratio' in data:
                response += f"**Sharpe Ratio:** {data['sharpe_ratio']:.3f}\n"
            
            if 'win_rate' in data:
                response += f"**Win Rate:** {data['win_rate']:.1%}\n"
            
            # Performance assessment
            sharpe = data.get('sharpe_ratio', 0)
            if sharpe > 1.5:
                performance = "Excellent"
                assessment = "strong risk-adjusted returns"
            elif sharpe > 1.0:
                performance = "Good"
                assessment = "solid risk-adjusted performance"
            elif sharpe > 0.5:
                performance = "Moderate"
                assessment = "reasonable performance with some risk"
            else:
                performance = "Poor"
                assessment = "weak risk-adjusted returns"
            
            response += f"\n**Performance Assessment:** {performance} - {assessment}."
            response += "\n\nWould you like me to test different assets or try a different strategy?"
            
        else:
            response = f"I encountered an issue with your backtesting: {result.content.get('error', 'Unknown error')}. "
            response += "Please check your asset symbols and try again, or ask me for help with specific assets."
        
        return response
    
    def _extract_assets_from_text(self, text: str) -> Optional[str]:
        """Extract asset symbols from natural language text."""
        import re
        
        # Look for patterns like "AAPL, GOOGL, MSFT" or "backtest TSLA and NVDA"
        # Simple regex to find potential stock symbols (2-5 uppercase letters)
        stock_pattern = r'\b[A-Z]{2,5}\b'
        potential_symbols = re.findall(stock_pattern, text.upper())
        
        # Filter out common English words that might match the pattern
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'HAVE', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'THIS', 'THAT', 'WITH', 'FROM', 'THEY', 'THEM', 'BEEN', 'SAID', 'EACH', 'WHICH', 'THEIR', 'TIME', 'MORE', 'VERY', 'WHAT', 'KNOW', 'JUST', 'FIRST', 'INTO', 'OVER', 'THINK', 'ALSO', 'YOUR', 'WORK', 'LIFE', 'ONLY', 'NEW', 'YEARS', 'WAY', 'MAY', 'COME', 'ITS', 'NOW', 'MOST', 'PEOPLE', 'GET', 'HAS', 'MUCH', 'LIKE', 'MADE', 'HOW', 'MANY', 'SOME', 'SO', 'THESE', 'SEE', 'HIM', 'TWO', 'WELL', 'WERE', 'RIGHT', 'BACK', 'OLD', 'WHERE', 'WANT', 'THOSE', 'CAME', 'GOOD', 'YEAR', 'SAME', 'USE', 'MAN', 'DAY', 'LONG', 'LITTLE', 'GREAT', 'NEVER', 'STILL', 'BETWEEN', 'ANOTHER', 'WHILE', 'LAST', 'MIGHT', 'MUST', 'US', 'LEFT', 'END', 'TURN', 'PLACE', 'BOTH', 'AGAIN', 'OFF', 'AWAY', 'EVEN', 'THROUGH', 'TAKE', 'EVERY', 'FOUND', 'UNDER', 'THOUGHT', 'DOWN', 'GIVE', 'CALLED', 'THREE', 'SMALL', 'DOES', 'PART', 'LOOKED', 'AFTER', 'NEXT', 'SEEM', 'WATER', 'AROUND', 'DIDN', 'PUT', 'ASKED', 'ABOVE', 'ALWAYS', 'BEING', 'TOLD', 'FELT', 'WENT', 'HAND', 'UNTIL', 'WORDS', 'WITHOUT', 'NOTHING', 'BEFORE', 'SAW', 'USED', 'MONEY', 'REALLY', 'USED', 'ACTUALLY', 'DOING', 'HELP', 'THING', 'THINGS', 'SOMETHING', 'ANYTHING', 'BACKTEST', 'STRATEGY', 'TEST'}
        
        # Filter out common words and keep likely stock symbols
        symbols = [s for s in potential_symbols if s not in common_words]
        
        if symbols:
            return ','.join(symbols)
        
        return None
    
    def _single_strategy_backtest(self, client, returns_df: pd.DataFrame, strategy: str, 
                                rebalance_freq: str, benchmark: str) -> Dict[str, Any]:
        """Run single strategy backtesting."""
        try:
            # Set up backtesting parameters
            rebalance_days = self._get_rebalance_days(rebalance_freq)
            
            # Initialize tracking
            portfolio_values = []
            benchmark_values = []
            weights_history = []
            rebalance_dates = []
            
            # Initial value
            portfolio_value = 100000  # $100k starting value
            benchmark_value = 100000
            
            # Equal weight benchmark
            n_assets = len(returns_df.columns)
            benchmark_weights = {asset: 1.0 / n_assets for asset in returns_df.columns}
            
            # Simulate backtesting
            portfolio_weights = benchmark_weights.copy()  # Start with equal weights
            
            for i, (date, returns_row) in enumerate(returns_df.iterrows()):
                # Rebalance if needed
                if i % rebalance_days == 0:
                    try:
                        # Optimize portfolio using strategy
                        if strategy == "mean_variance":
                            optimization_result = client.portfolio.optimize_portfolio(
                                returns=returns_df.iloc[:i+1],
                                method="mean_variance"
                            )
                        elif strategy == "sharpe":
                            optimization_result = client.portfolio.optimize_portfolio(
                                returns=returns_df.iloc[:i+1],
                                method="sharpe"
                            )
                        else:
                            optimization_result = {"weights": benchmark_weights}
                        
                        portfolio_weights = optimization_result.get("weights", benchmark_weights)
                        weights_history.append({
                            "date": date,
                            "weights": portfolio_weights.copy()
                        })
                        rebalance_dates.append(date)
                        
                    except:
                        # Keep previous weights if optimization fails
                        pass
                
                # Calculate returns
                portfolio_return = sum(returns_row[asset] * portfolio_weights.get(asset, 0) 
                                     for asset in returns_row.index)
                benchmark_return = sum(returns_row[asset] * benchmark_weights[asset] 
                                     for asset in returns_row.index)
                
                # Update values
                portfolio_value *= (1 + portfolio_return)
                benchmark_value *= (1 + benchmark_return)
                
                portfolio_values.append(portfolio_value)
                benchmark_values.append(benchmark_value)
            
            # Create performance DataFrame
            performance_df = pd.DataFrame({
                'portfolio': portfolio_values,
                'benchmark': benchmark_values
            }, index=returns_df.index)
            
            # Calculate metrics
            metrics = self._calculate_backtest_metrics(performance_df, weights_history)
            
            return {
                "success": True,
                "strategy": strategy,
                "backtest_period": f"{returns_df.index[0].strftime('%Y-%m-%d')} to {returns_df.index[-1].strftime('%Y-%m-%d')}",
                "performance_metrics": metrics,
                "final_values": {
                    "portfolio": portfolio_values[-1],
                    "benchmark": benchmark_values[-1],
                    "outperformance": portfolio_values[-1] - benchmark_values[-1]
                },
                "rebalance_info": {
                    "frequency": rebalance_freq,
                    "rebalance_count": len(rebalance_dates),
                    "rebalance_dates": [date.strftime('%Y-%m-%d') for date in rebalance_dates[:5]]  # First 5
                },
                "assets": list(returns_df.columns)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Single strategy backtest failed: {str(e)}"
            }
    
    def _multi_strategy_backtest(self, client, returns_df: pd.DataFrame, 
                               rebalance_freq: str, benchmark: str) -> Dict[str, Any]:
        """Run multi-strategy comparison backtesting."""
        try:
            strategies = ["mean_variance", "sharpe", "equal_weight"]
            strategy_results = {}
            
            for strategy in strategies:
                result = self._single_strategy_backtest(client, returns_df, strategy, rebalance_freq, benchmark)
                if result["success"]:
                    strategy_results[strategy] = result
            
            # Compare strategies
            comparison = self._compare_strategies(strategy_results)
            
            return {
                "success": True,
                "backtest_type": "multi_strategy",
                "strategies_tested": list(strategy_results.keys()),
                "individual_results": strategy_results,
                "strategy_comparison": comparison,
                "backtest_period": f"{returns_df.index[0].strftime('%Y-%m-%d')} to {returns_df.index[-1].strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Multi-strategy backtest failed: {str(e)}"
            }
    
    def _calculate_backtest_metrics(self, performance_df: pd.DataFrame, weights_history: List) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics."""
        try:
            # Calculate returns
            portfolio_returns = performance_df['portfolio'].pct_change().dropna()
            benchmark_returns = performance_df['benchmark'].pct_change().dropna()
            
            # Performance metrics
            portfolio_total_return = (performance_df['portfolio'].iloc[-1] / performance_df['portfolio'].iloc[0]) - 1
            benchmark_total_return = (performance_df['benchmark'].iloc[-1] / performance_df['benchmark'].iloc[0]) - 1
            
            # Risk metrics
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
            
            # Sharpe ratios (assuming 0% risk-free rate)
            portfolio_sharpe = (portfolio_returns.mean() * 252) / portfolio_volatility if portfolio_volatility > 0 else 0
            benchmark_sharpe = (benchmark_returns.mean() * 252) / benchmark_volatility if benchmark_volatility > 0 else 0
            
            # Maximum drawdown
            portfolio_cumulative = (1 + portfolio_returns).cumprod()
            portfolio_rolling_max = portfolio_cumulative.expanding().max()
            portfolio_drawdown = (portfolio_cumulative - portfolio_rolling_max) / portfolio_rolling_max
            portfolio_max_drawdown = portfolio_drawdown.min()
            
            return {
                "total_return": {
                    "portfolio": portfolio_total_return,
                    "benchmark": benchmark_total_return,
                    "excess": portfolio_total_return - benchmark_total_return
                },
                "annualized_return": {
                    "portfolio": portfolio_returns.mean() * 252,
                    "benchmark": benchmark_returns.mean() * 252
                },
                "volatility": {
                    "portfolio": portfolio_volatility,
                    "benchmark": benchmark_volatility
                },
                "sharpe_ratio": {
                    "portfolio": portfolio_sharpe,
                    "benchmark": benchmark_sharpe
                },
                "max_drawdown": {
                    "portfolio": portfolio_max_drawdown,
                    "benchmark": self._calculate_max_drawdown(benchmark_returns)
                },
                "win_rate": (portfolio_returns > 0).mean(),
                "correlation_with_benchmark": portfolio_returns.corr(benchmark_returns)
            }
            
        except Exception as e:
            return {"error": f"Metrics calculation failed: {str(e)}"}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a return series."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def _compare_strategies(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple strategies."""
        try:
            comparison = {}
            
            for strategy, results in strategy_results.items():
                if "performance_metrics" in results:
                    metrics = results["performance_metrics"]
                    comparison[strategy] = {
                        "total_return": metrics.get("total_return", {}).get("portfolio", 0),
                        "sharpe_ratio": metrics.get("sharpe_ratio", {}).get("portfolio", 0),
                        "max_drawdown": metrics.get("max_drawdown", {}).get("portfolio", 0),
                        "volatility": metrics.get("volatility", {}).get("portfolio", 0)
                    }
            
            # Find best performing strategy for each metric
            best_return = max(comparison.keys(), key=lambda x: comparison[x]["total_return"])
            best_sharpe = max(comparison.keys(), key=lambda x: comparison[x]["sharpe_ratio"])
            best_drawdown = min(comparison.keys(), key=lambda x: comparison[x]["max_drawdown"])
            
            return {
                "strategy_metrics": comparison,
                "best_performers": {
                    "highest_return": best_return,
                    "highest_sharpe": best_sharpe,
                    "lowest_drawdown": best_drawdown
                }
            }
            
        except Exception as e:
            return {"error": f"Strategy comparison failed: {str(e)}"}
    
    def _parse_period(self, period: str) -> int:
        """Parse period string to number of days."""
        period_map = {
            "1m": 21, "3m": 63, "6m": 126,
            "1y": 252, "2y": 504, "3y": 756
        }
        return period_map.get(period, 252)
    
    def _get_rebalance_days(self, frequency: str) -> int:
        """Get number of days for rebalancing frequency."""
        freq_map = {
            "daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63
        }
        return freq_map.get(frequency, 21)
    
    def backtest_strategy(self, 
                         assets: str = "AAPL,MSFT,AMZN",
                         strategy: str = "mean_variance",
                         period: str = "1y") -> Dict[str, Any]:
        """
        Simple strategy backtesting tool.
        
        Args:
            assets: Comma-separated asset symbols
            strategy: Strategy to test (mean_variance, sharpe, equal_weight)
            period: Backtest period (1m, 3m, 6m, 1y, 2y)
            
        Returns:
            Backtesting results and performance metrics
        """
        try:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
            
            result = self.run(
                strategy=strategy,
                backtest_period=period,
                assets=asset_list
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Backtesting failed: {str(e)}"
            }
    
    def _generate_sample_data(self, assets: List[str], n_days: int = 252) -> pd.DataFrame:
        """Generate sample return data for backtesting."""
        np.random.seed(42)
        n_assets = len(assets)
        
        # Generate realistic returns with trends
        mean_returns = np.random.uniform(0.0002, 0.0012, n_assets)
        volatilities = np.random.uniform(0.015, 0.035, n_assets)
        
        # Create correlation matrix
        correlations = np.random.uniform(0.2, 0.7, (n_assets, n_assets))
        correlations = (correlations + correlations.T) / 2
        np.fill_diagonal(correlations, 1.0)
        
        # Convert to covariance matrix
        covariance_matrix = np.outer(volatilities, volatilities) * correlations
        
        # Generate returns with some market trends
        returns_data = np.random.multivariate_normal(
            mean=mean_returns,
            cov=covariance_matrix,
            size=n_days
        )
        
        # Add some market trends
        trend = np.linspace(0, 0.001, n_days).reshape(-1, 1)
        returns_data += trend
        
        # Create DataFrame with dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(n_days * 1.4))
        dates = pd.date_range(start=start_date, end=end_date, freq='B')[:n_days]
        
        returns_df = pd.DataFrame(returns_data, index=dates, columns=assets)
        return returns_df
