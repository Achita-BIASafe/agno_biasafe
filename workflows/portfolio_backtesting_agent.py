"""
Portfolio Backtesting Agent for Agno Framework
"""

from agno.agent import Agent
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for api_config import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from api_config import get_endpoint
from .portfolio_backtester import PortfolioBacktester

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


class PortfolioBacktestingAgent(Agent):
    """Portfolio backtesting agent for conversational strategy testing."""
    
    name = "Portfolio Backtester"
    role = "Strategy Testing & Performance Analysis Specialist"
    
    instructions = [
        "You are an expert portfolio backtesting agent that helps users test investment strategies.",
        "You can backtest custom asset selections and provide comprehensive performance analysis.",
        "You specialize in historical performance analysis, drawdown analysis, and strategy comparison.",
        "Always explain backtesting results and their implications for strategy effectiveness.",
        "Provide clear recommendations based on historical performance data.",
        "When users mention specific stock symbols, extract them and backtest strategies.",
        "Help users understand the difference between historical and future performance.",
        "Provide actionable insights for strategy improvement and risk management."
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backtester = PortfolioBacktester()
    
    def run(self, message: str, **kwargs):
        """
        Process user message and provide backtesting insights.
        
        Args:
            message: User's natural language input about backtesting
            **kwargs: Additional arguments from the Agno framework (session_id, user_id, etc.)
            
        Returns:
            String response with conversational backtesting results
        """
        try:
            # Use the conversational handler from the workflow
            response_text = self.backtester.handle_conversation(message)
            
            # Return as plain string
            return str(response_text)
            
        except Exception as e:
            error_message = f"I encountered an issue while backtesting your strategy: {str(e)}. Please try rephrasing your question or ask me for help with specific assets."
            return str(error_message)
    
    def backtest_strategy(self, assets: str, strategy: str = "equal_weight", period: str = "1Y") -> str:
        """
        Direct backtesting method for specific assets and strategy.
        
        Args:
            assets: Comma-separated asset symbols
            strategy: Strategy to test (equal_weight, momentum, min_variance)
            period: Backtesting period
            
        Returns:
            Formatted backtesting results
        """
        try:
            result = self.backtester.run(assets=assets, strategy=strategy, backtest_period=period)
            
            if result.content.get("success"):
                response = f"📈 **Backtesting Results**\n\n"
                response += f"**Assets:** {assets}\n"
                response += f"**Strategy:** {strategy.replace('_', ' ').title()}\n"
                response += f"**Period:** {period}\n\n"
                
                data = result.content
                
                # Performance metrics
                response += "**🎯 Performance Metrics:**\n"
                if 'total_return' in data:
                    response += f"• Total Return: {data['total_return']:.2%}\n"
                
                if 'annualized_return' in data:
                    response += f"• Annualized Return: {data['annualized_return']:.2%}\n"
                
                if 'volatility' in data:
                    response += f"• Volatility: {data['volatility']:.2%}\n"
                
                if 'max_drawdown' in data:
                    response += f"• Maximum Drawdown: {data['max_drawdown']:.2%}\n"
                
                if 'sharpe_ratio' in data:
                    response += f"• Sharpe Ratio: {data['sharpe_ratio']:.3f}\n"
                
                if 'win_rate' in data:
                    response += f"• Win Rate: {data['win_rate']:.1%}\n"
                
                # Performance assessment
                sharpe = data.get('sharpe_ratio', 0)
                if sharpe > 1.5:
                    assessment = "🟢 **Excellent Performance** - Strong risk-adjusted returns"
                elif sharpe > 1.0:
                    assessment = "🟡 **Good Performance** - Solid risk-adjusted performance"
                elif sharpe > 0.5:
                    assessment = "🟠 **Moderate Performance** - Reasonable performance with some risk"
                else:
                    assessment = "🔴 **Poor Performance** - Weak risk-adjusted returns"
                
                response += f"\n**📊 Assessment:**\n{assessment}\n"
                
                # Recommendations
                response += "\n**💡 Recommendations:**\n"
                max_dd = data.get('max_drawdown', 0)
                if max_dd > 0.2:
                    response += "• Consider risk management measures due to high drawdown\n"
                if sharpe < 1.0:
                    response += "• Explore strategy optimization or different asset allocation\n"
                if data.get('volatility', 0) > 0.25:
                    response += "• High volatility - consider position sizing adjustments\n"
                
                response += "• Remember: Past performance doesn't guarantee future results\n"
                
                return response
            else:
                return f"❌ Backtesting failed: {result.content.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"❌ Error during backtesting: {str(e)}"
