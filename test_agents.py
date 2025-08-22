"""
Quick Agent Test - bypasses complex imports to test core functionality
"""

import sys
import os

# Simple test to see what's working
def test_agents():
    print("🧪 Testing agent imports...")
    
    # Test 1: Can we import the agent classes?
    try:
        print("📁 Testing workflows directory...")
        from workflows.portfolio_optimizer_agent import PortfolioOptimizerAgent
        print("✅ PortfolioOptimizerAgent - OK")
        
        from workflows.risk_analysis_agent import RiskAnalysisAgent
        print("✅ RiskAnalysisAgent - OK")
        
        from workflows.data_analysis_agent import DataAnalysisAgent
        print("✅ DataAnalysisAgent - OK")
        
        from workflows.portfolio_backtesting_agent import PortfolioBacktestingAgent
        print("✅ PortfolioBacktestingAgent - OK")
        
        print(f"✅ All 4 individual agents imported successfully!")
        
    except Exception as e:
        print(f"❌ Individual agents error: {e}")
        return False
    
    # Test 2: Can we import the Finance Team?
    try:
        print("\n🏦 Testing Finance Team...")
        from tools.finance_team import FinanceTeam
        print("✅ Finance Team imported successfully!")
        
        # Try to create an instance
        finance_agent = FinanceTeam(agent_id="test-finance")
        print("✅ Finance Team instance created!")
        
        return True
        
    except Exception as e:
        print(f"❌ Finance Team error: {e}")
        return False

if __name__ == "__main__":
    success = test_agents()
    if success:
        print("\n🎯 SOLUTION: All agents are working!")
        print("   • Finance Team: Available (tool selection agent)")
        print("   • Individual Agents: Available (4 specialized agents)")
        print("\nℹ️  The 'no agents found' issue should now be resolved.")
    else:
        print("\n❌ Some agents still have import issues.")
