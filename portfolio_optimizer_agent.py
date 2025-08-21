"""
Portfolio Optimization Agent for Agno Framework
"""

from agno.agent import Agent
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json
import requests
from api_config import get_endpoint
from portfolio_optimizer import PortfolioOptimizer

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


class PortfolioOptimizerAgent(Agent):
    """Portfolio optimization agent for conversational financial analysis."""
    
    name = "Portfolio Optimizer"
    role = "Expert Portfolio Manager & Risk Analyst"
    
    instructions = [
        "You are an expert portfolio optimization agent that helps users create optimal investment portfolios.",
        "You can analyze custom asset selections and provide portfolio recommendations.",
        "You use advanced risk measures and optimization techniques through the riskfolio engine.",
        "Always ask for clarification if the user's portfolio requirements are unclear.",
        "Provide clear explanations of your optimization results and recommendations.",
        "When users mention specific stock symbols, extract them and use them for optimization.",
        "Explain the results in simple terms that non-experts can understand.",
        "Always provide actionable insights and recommendations."
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = PortfolioOptimizer()
    
    def run(self, message: str, **kwargs):
        """
        Process user message and provide portfolio optimization insights.
        
        Args:
            message: User's natural language input about portfolio optimization
            **kwargs: Additional arguments from the Agno framework (session_id, user_id, etc.)
            
        Returns:
            String response with conversational portfolio recommendations
        """
        try:
            # Use the conversational handler from the workflow
            response_text = self.optimizer.handle_conversation(message)
            
            # Return as plain string instead of RunResponse
            return str(response_text)
            
        except Exception as e:
            error_message = f"I encountered an issue while analyzing your portfolio request: {str(e)}. Please try rephrasing your question or ask me for help with specific assets."
            return str(error_message)
    
    def extract_assets_and_optimize(self, assets: str, strategy: str = "MinRisk") -> str:
        """
        Direct optimization method for specific assets.
        
        Args:
            assets: Comma-separated asset symbols
            strategy: Optimization strategy
            
        Returns:
            Formatted optimization results
        """
        try:
            result = self.optimizer.run(assets=assets)
            
            if result.content.get("success"):
                response = f"✅ **Portfolio Optimization Complete**\n\n"
                response += f"**Assets Analyzed:** {assets}\n"
                response += f"**Strategy:** {strategy}\n\n"
                
                data = result.content
                if 'weights' in data:
                    response += "**Optimal Portfolio Allocation:**\n"
                    for asset, weight in data['weights'].items():
                        response += f"• {asset}: {weight:.2%}\n"
                
                if 'expected_return' in data:
                    response += f"\n📈 **Expected Annual Return:** {data['expected_return']:.2%}"
                
                if 'volatility' in data:
                    response += f"\n📊 **Expected Volatility:** {data['volatility']:.2%}"
                
                if 'sharpe_ratio' in data:
                    response += f"\n⚡ **Sharpe Ratio:** {data['sharpe_ratio']:.3f}"
                
                return response
            else:
                return f"❌ Optimization failed: {result.content.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"❌ Error during optimization: {str(e)}"
