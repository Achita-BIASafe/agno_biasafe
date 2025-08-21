"""
Test script to understand how agno workflows expose tools
"""

from agno.workflow import Workflow
from typing import Dict, Any

class TestWorkflow(Workflow):
    """Test workflow to understand tool exposure"""
    
    description = "Test workflow for understanding agno tool patterns"
    
    def run(self, message: str = "Hello World") -> Dict[str, Any]:
        """Main run method"""
        return {"message": message, "status": "success"}
    
    def calculate_portfolio(self, assets: list = None, risk_level: str = "medium") -> Dict[str, Any]:
        """Calculate portfolio optimization"""
        return {
            "assets": assets or ["AAPL", "MSFT"],
            "risk_level": risk_level,
            "weights": {"AAPL": 0.6, "MSFT": 0.4}
        }
    
    def analyze_risk(self, portfolio: dict = None) -> Dict[str, Any]:
        """Analyze portfolio risk"""
        return {
            "var": 0.05,
            "sharpe": 1.2,
            "volatility": 0.15
        }

if __name__ == "__main__":
    # Test the workflow
    workflow = TestWorkflow()
    print("Workflow methods:")
    for method in dir(workflow):
        if not method.startswith('_') and callable(getattr(workflow, method)):
            print(f"  {method}")
    
    # Check if workflow has tools attribute
    if hasattr(workflow, 'tools'):
        print(f"Tools: {workflow.tools}")
    else:
        print("No tools attribute found")
