"""
BiasafeAI Agno Playground
1. Install dependencies using: `pip install agno pandas numpy requests fastapi uvicorn sqlalchemy riskfolio-lib python-dotenv`
2. Run the script using: `python playground.py`
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from agno.playground import Playground
from agno.storage.sqlite import SqliteStorage

# Import the workflows
from portfolio_optimizer import PortfolioOptimizer
from risk_analyzer import RiskAnalyzer  
from portfolio_backtester import PortfolioBacktester
from data_analyzer import DataAnalyzer
from plotting_tools import PlottingTools
from report_generator import ReportGenerator
from data_utilities import DataUtilities

# Import the agents
try:
    from portfolio_optimizer_agent import PortfolioOptimizerAgent
    from risk_analysis_agent import RiskAnalysisAgent
    from data_analysis_agent import DataAnalysisAgent
    from portfolio_backtesting_agent import PortfolioBacktestingAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agents: {e}")
    AGENTS_AVAILABLE = False

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
        
        agents = [
            portfolio_optimizer_agent,
            risk_analysis_agent,
            data_analysis_agent,
            portfolio_backtesting_agent,
        ]
        
        print(f"✅ Initialized {len(agents)} conversational agents")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize agents: {e}")
        agents = []

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
    name="BiasafeAI Portfolio Optimization Playground",
)
app = playground.get_app(use_async=False)

if __name__ == "__main__":
    # Start the playground server
    playground.serve(
        app="playground:app",
        host="localhost",
        port=7777,
        reload=True,
    )
