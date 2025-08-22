"""
Risk Analysis Agent for Agno Framework
"""

from agno.agent import Agent
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import requests
import sys
import os

# Add parent directory to path for api_config import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from api_config import get_endpoint
from .risk_analyzer import RiskAnalyzer

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


class RiskAnalysisAgent(Agent):
    """Risk analysis agent for conversational portfolio risk assessment."""
    
    name = "Risk Analyzer"
    role = "Portfolio Risk Assessment Specialist"
    
    instructions = [
        "You are an expert risk analysis agent that helps users understand portfolio risk metrics.",
        "You can analyze custom asset selections and provide comprehensive risk assessments.",
        "You specialize in volatility analysis, risk-adjusted returns, and downside protection metrics.",
        "Always explain risk metrics in simple terms that users can understand.",
        "Provide actionable insights about portfolio risk levels and recommendations.",
        "When users mention specific stock symbols, extract them and analyze their risk profile.",
        "Help users understand the trade-offs between risk and return.",
        "Provide clear guidance on risk management strategies."
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = RiskAnalyzer()
    
    def run(self, message: str, **kwargs):
        """
        Process user message and provide risk analysis insights.
        
        Args:
            message: User's natural language input about risk analysis
            **kwargs: Additional arguments from the Agno framework (session_id, user_id, etc.)
            
        Returns:
            String response with conversational risk assessment
        """
        try:
            # Use the conversational handler from the workflow
            response_text = self.analyzer.handle_conversation(message)
            
            # Return as plain string
            return str(response_text)
            
        except Exception as e:
            error_message = f"I encountered an issue while analyzing the risk profile: {str(e)}. Please try rephrasing your question or ask me for help with specific assets."
            return str(error_message)
    
    def analyze_portfolio_risk(self, assets: str, weights: str = None) -> str:
        """
        Direct risk analysis method for specific assets.
        
        Args:
            assets: Comma-separated asset symbols
            weights: Portfolio weights (optional)
            
        Returns:
            Formatted risk analysis results
        """
        try:
            if weights:
                result = self.analyzer.run(assets=assets, portfolio_weights=weights)
            else:
                result = self.analyzer.run(assets=assets)
            
            if result.content.get("success"):
                response = f"🛡️ **Risk Analysis Complete**\n\n"
                response += f"**Assets Analyzed:** {assets}\n"
                if weights:
                    response += f"**Weights:** {weights}\n"
                response += "\n"
                
                data = result.content
                if 'portfolio_volatility' in data:
                    response += f"📊 **Portfolio Volatility:** {data['portfolio_volatility']:.2%}\n"
                
                if 'var_95' in data:
                    response += f"⚠️ **Value at Risk (95%):** {data['var_95']:.2%}\n"
                
                if 'cvar_95' in data:
                    response += f"🔴 **Conditional VaR (95%):** {data['cvar_95']:.2%}\n"
                
                if 'max_drawdown' in data:
                    response += f"📉 **Maximum Drawdown:** {data['max_drawdown']:.2%}\n"
                
                if 'sharpe_ratio' in data:
                    response += f"⚡ **Sharpe Ratio:** {data['sharpe_ratio']:.3f}\n"
                
                # Risk level assessment
                volatility = data.get('portfolio_volatility', 0)
                if volatility < 0.15:
                    risk_level = "🟢 Low Risk"
                    risk_desc = "Conservative portfolio with steady returns expected"
                elif volatility < 0.25:
                    risk_level = "🟡 Moderate Risk"
                    risk_desc = "Balanced portfolio with reasonable volatility"
                else:
                    risk_level = "🔴 High Risk"
                    risk_desc = "Aggressive portfolio with potential for large swings"
                
                response += f"\n**Risk Assessment:** {risk_level}\n{risk_desc}"
                
                return response
            else:
                return f"❌ Risk analysis failed: {result.content.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"❌ Error during risk analysis: {str(e)}"
