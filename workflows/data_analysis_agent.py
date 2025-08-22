"""
Data Analysis Agent for Agno Framework
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
from .data_analyzer import DataAnalyzer

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


class DataAnalysisAgent(Agent):
    """Data analysis agent for conversational financial market analysis."""
    
    name = "Data Analyst"
    role = "Financial Data Analyst & Market Research Specialist"
    
    instructions = [
        "You are an expert financial data analysis agent that helps users understand market trends and patterns.",
        "You can analyze custom asset selections and provide comprehensive market data insights.",
        "You specialize in correlation analysis, return distributions, and historical performance metrics.",
        "Always provide clear explanations of statistical findings in accessible language.",
        "Offer actionable insights about market conditions and asset relationships.",
        "When users mention specific stock symbols, extract them and analyze their data patterns.",
        "Help users understand market correlations and diversification benefits.",
        "Provide data-driven insights for investment decision making."
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = DataAnalyzer()
    
    def run(self, message: str, **kwargs):
        """
        Process user message and provide data analysis insights.
        
        Args:
            message: User's natural language input about data analysis
            **kwargs: Additional arguments from the Agno framework (session_id, user_id, etc.)
            
        Returns:
            String response with conversational data insights
        """
        try:
            # Use the conversational handler from the workflow
            response_text = self.analyzer.handle_conversation(message)
            
            # Return as plain string
            return str(response_text)
            
        except Exception as e:
            error_message = f"I encountered an issue while analyzing the market data: {str(e)}. Please try rephrasing your question or ask me for help with specific assets."
            return str(error_message)
    
    def analyze_market_data(self, assets: str, analysis_type: str = "overview") -> str:
        """
        Direct data analysis method for specific assets.
        
        Args:
            assets: Comma-separated asset symbols
            analysis_type: Type of analysis (overview, correlation, quality)
            
        Returns:
            Formatted data analysis results
        """
        try:
            result = self.analyzer.run(assets=assets, analysis_type=analysis_type)
            
            if result.content.get("success"):
                response = f"📊 **Market Data Analysis Complete**\n\n"
                response += f"**Assets Analyzed:** {assets}\n"
                response += f"**Analysis Type:** {analysis_type.title()}\n\n"
                
                data = result.content
                
                if analysis_type == "correlation" and 'correlation_matrix' in data:
                    response += "**📈 Correlation Analysis:**\n"
                    correlations = data['correlation_matrix']
                    # Find highest correlation
                    if correlations:
                        max_corr = max([max(row.values()) for row in correlations.values() if row])
                        response += f"• Highest correlation: {max_corr:.2%}\n"
                        response += "• Strong correlations indicate assets move together\n"
                        response += "• Consider diversification to reduce correlation risk\n"
                        
                elif analysis_type == "quality" and 'data_quality' in data:
                    response += "**🔍 Data Quality Assessment:**\n"
                    quality = data['data_quality']
                    if 'completeness' in quality:
                        response += f"• Data completeness: {quality['completeness']:.1%}\n"
                    if 'outliers_detected' in quality:
                        response += f"• Outliers detected: {quality['outliers_detected']}\n"
                    response += "• Clean data ensures reliable analysis results\n"
                        
                else:  # overview analysis
                    if 'asset_metrics' in data:
                        response += "**📈 Market Overview:**\n"
                        metrics = data['asset_metrics']
                        
                        # Find best and worst performers
                        best_performer = max(metrics.keys(), key=lambda x: metrics[x].get('total_return', 0))
                        worst_performer = min(metrics.keys(), key=lambda x: metrics[x].get('total_return', 0))
                        
                        response += f"• 🚀 Best performer: {best_performer} ({metrics[best_performer].get('total_return', 0):.2%})\n"
                        response += f"• 📉 Worst performer: {worst_performer} ({metrics[worst_performer].get('total_return', 0):.2%})\n"
                        
                        avg_volatility = np.mean([m.get('volatility', 0) for m in metrics.values()])
                        response += f"• 📊 Average volatility: {avg_volatility:.2%}\n"
                        
                        response += "\n**💡 Investment Insights:**\n"
                        response += "• Diversify across top performers for balanced risk\n"
                        response += "• Monitor volatility levels for risk management\n"
                        response += "• Consider rebalancing based on performance trends\n"
                
                return response
            else:
                return f"❌ Data analysis failed: {result.content.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"❌ Error during data analysis: {str(e)}"
