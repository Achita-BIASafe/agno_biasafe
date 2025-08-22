"""
BiasafeAI Agno Playground - Team-Based Architecture

Features:
• NEW: Finance Team - Unified conversational agent with individual callable tools
• Legacy: Individual agents for backward compatibility
• Workflows: Portfolio optimization, risk analysis, backtesting, data analysis

Architecture:
• Finance Team routes requests to appropriate specialized tools
• Each financial analysis function available as individual tool for LLMs
• Direct tool access + conversational interface

Usage:
1. Install dependencies: `pip install agno pandas numpy requests fastapi uvicorn sqlalchemy riskfolio-lib python-dotenv`
2. Set OpenAI API key in .env file: OPENAI_API_KEY=your_key_here  
3. Run: `python playground.py`
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from agno.playground import Playground
from agno.storage.sqlite import SqliteStorage

# Import the workflows
from workflows.portfolio_optimizer import PortfolioOptimizer
from workflows.risk_analyzer import RiskAnalyzer  
from workflows.portfolio_backtester import PortfolioBacktester
from workflows.data_analyzer import DataAnalyzer
from workflows.plotting_tools import PlottingTools
from workflows.report_generator import ReportGenerator
from workflows.data_utilities import DataUtilities

# Import the agents
try:
    from workflows.portfolio_optimizer_agent import PortfolioOptimizerAgent
    from workflows.risk_analysis_agent import RiskAnalysisAgent
    from workflows.data_analysis_agent import DataAnalysisAgent
    from workflows.portfolio_backtesting_agent import PortfolioBacktestingAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agents: {e}")
    AGENTS_AVAILABLE = False

# Import the new Finance Team
try:
    from tools.finance_team import FinanceTeam
    FINANCE_TEAM_AVAILABLE = True
    print("✅ New Finance Team architecture available")
except ImportError as e:
    print(f"Warning: Could not import Finance Team: {e}")
    FINANCE_TEAM_AVAILABLE = False

# Initialize the workflows with SQLite storage

portfolio_optimizer = PortfolioOptimizer(
    workflow_id="portfolio-optimization",
    storage=SqliteStorage(
        table_name="portfolio_optimization_workflows",
        db_file="tmp/agno_workflows.db",
        mode="workflow",
        auto_upgrade_schema=True,
    ),
)

risk_analyzer = RiskAnalyzer(
    workflow_id="risk-analysis",
    storage=SqliteStorage(
        table_name="risk_analysis_workflows",
        db_file="tmp/agno_workflows.db",
        mode="workflow",
        auto_upgrade_schema=True,
    ),
)

portfolio_backtester = PortfolioBacktester(
    workflow_id="portfolio-backtesting",
    storage=SqliteStorage(
        table_name="portfolio_backtesting_workflows",
        db_file="tmp/agno_workflows.db",
        mode="workflow",
        auto_upgrade_schema=True,
    ),
)

data_analyzer = DataAnalyzer(
    workflow_id="data-analysis",
    storage=SqliteStorage(
        table_name="data_analysis_workflows",
        db_file="tmp/agno_workflows.db",
        mode="workflow",
        auto_upgrade_schema=True,
    ),
)

plotting_tools = PlottingTools(
    workflow_id="plotting-tools",
    storage=SqliteStorage(
        table_name="plotting_tools_workflows",
        db_file="tmp/agno_workflows.db",
        mode="workflow",
        auto_upgrade_schema=True,
    ),
)

report_generator = ReportGenerator(
    workflow_id="report-generator",
    storage=SqliteStorage(
        table_name="report_generator_workflows",
        db_file="tmp/agno_workflows.db",
        mode="workflow",
        auto_upgrade_schema=True,
    ),
)

data_utilities = DataUtilities(
    workflow_id="data-utilities",
    storage=SqliteStorage(
        table_name="data_utilities_workflows",
        db_file="tmp/agno_workflows.db",
        mode="workflow",
        auto_upgrade_schema=True,
    ),
)

# Initialize agents if available
agents = []

# Add the new Finance Team (preferred architecture)
if FINANCE_TEAM_AVAILABLE:
    try:
        finance_team = FinanceTeam(
            agent_id="finance-team",
            storage=SqliteStorage(
                table_name="finance_team_sessions",
                db_file="tmp/agno_agents.db",
                mode="agent",
                auto_upgrade_schema=True,
            ),
        )
        agents.append(finance_team)
        
        # Show available tools
        from tools.finance_tools import list_tools, get_tool_info
        tools = list_tools()
        tool_info = get_tool_info()
        
        print("✅ Finance Team initialized with individual tools:")
        for category, tool_list in tool_info["categories"].items():
            print(f"   📊 {category}: {len(tool_list)} tools")
        print(f"   Total: {len(tools)} specialized financial analysis tools")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize Finance Team: {e}")

# Add legacy individual agents (backward compatibility)
if AGENTS_AVAILABLE:
    try:
        portfolio_optimizer_agent = PortfolioOptimizerAgent(
            agent_id="portfolio-optimizer-agent",
            storage=SqliteStorage(
                table_name="portfolio_optimizer_agents",
                db_file="tmp/agno_agents.db",
                mode="agent",
                auto_upgrade_schema=True,
            ),
        )
        
        risk_analysis_agent = RiskAnalysisAgent(
            agent_id="risk-analysis-agent",
            storage=SqliteStorage(
                table_name="risk_analysis_agents",
                db_file="tmp/agno_agents.db",
                mode="agent",
                auto_upgrade_schema=True,
            ),
        )
        
        data_analysis_agent = DataAnalysisAgent(
            agent_id="data-analysis-agent",
            storage=SqliteStorage(
                table_name="data_analysis_agents",
                db_file="tmp/agno_agents.db",
                mode="agent",
                auto_upgrade_schema=True,
            ),
        )
        
        portfolio_backtesting_agent = PortfolioBacktestingAgent(
            agent_id="portfolio-backtesting-agent",
            storage=SqliteStorage(
                table_name="portfolio_backtesting_agents",
                db_file="tmp/agno_agents.db",
                mode="agent",
                auto_upgrade_schema=True,
            ),
        )
        
        # Add legacy agents to the list
        legacy_agents = [
            portfolio_optimizer_agent,
            risk_analysis_agent,
            data_analysis_agent,
            portfolio_backtesting_agent,
        ]
        
        agents.extend(legacy_agents)
        print(f"✅ Initialized {len(legacy_agents)} legacy individual agents (backward compatibility)")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize legacy agents: {e}")

print(f"🎯 Total agents available: {len(agents)}")
if FINANCE_TEAM_AVAILABLE:
    print("   • Finance Team (NEW) - Unified agent with individual callable tools")
if AGENTS_AVAILABLE:
    print("   • Legacy Individual Agents - Portfolio, Risk, Data Analysis, Backtesting")

# Initialize the Playground with workflows and agents
playground = Playground(
    workflows=[
        portfolio_optimizer,
        risk_analyzer,
        portfolio_backtester,
        data_analyzer,
        plotting_tools,
        report_generator,
        data_utilities,
    ],
    agents=agents,
    app_id="biasafe-ai-playground",
    name="BiasafeAI Financial Analysis Playground - Team-Based Architecture",
)
app = playground.get_app(use_async=False)

if __name__ == "__main__":
    print("=" * 80)
    print("🏦 BIASAFEAI FINANCIAL ANALYSIS PLAYGROUND")
    print("=" * 80)
    print()
    print("🎯 NEW TEAM-BASED ARCHITECTURE:")
    if FINANCE_TEAM_AVAILABLE:
        print("   ✅ Finance Team - Unified agent with individual callable tools")
        print("   🛠️ Individual financial analysis tools for direct LLM access")
    if AGENTS_AVAILABLE:
        print("   🔄 Legacy individual agents (backward compatibility)")
    print()
    print("📊 Available Analysis Types:")
    print("   • Portfolio Optimization & Rebalancing")
    print("   • Risk Analysis & Stress Testing") 
    print("   • Performance Attribution & Benchmarking")
    print("   • Strategy Backtesting & Validation")
    print("   • Market Data Analysis & Correlation")
    print("   • Report Generation & Visualization")
    print()
    print("🌐 Starting playground server...")
    print("=" * 80)
    
    # Start the playground server
    playground.serve(
        app="playground:app",
        host="localhost",
        port=7777,
        reload=True,
    )
